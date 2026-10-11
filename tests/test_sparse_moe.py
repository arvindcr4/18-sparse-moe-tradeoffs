"""Correctness tests for the sparse-MoE trade-off lab."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sparse_moe_tradeoffs import (IGNORE_INDEX, CharData, CharLM, MoELayer,
                                  MoESpec, active_expert_param_count,
                                  build_corpus, expert_param_count,
                                  make_val_batches, matched_family,
                                  train_model, variant)
from sparse_moe_tradeoffs.configs import active_total_ratio

D = 96


def small_layer(**over):
    """A small controllable MoE layer (d=8, N=4, k=2)."""
    torch.manual_seed(0)
    defaults = dict(n_experts=4, k=2, expert_hidden=5, shared_hidden=0,
                    balance="none")
    defaults.update(over)
    return MoELayer(8, MoESpec(**defaults))


def tiny_data(block_size=4):
    text = "abcdefghijklmnopqrstuvwxyz" * 8
    tags = np.zeros(len(text), dtype=np.int8)
    return CharData(text, tags, block_size=block_size, val_frac=0.2)


# ---------------------------------------------------------------- parameters

def test_param_matching_closed_form_and_torch_counts():
    """Coarse (2 big, top-1) and flagship (8 fine + shared, top-2) match on
    total AND active expert params — closed form and actual torch counts."""
    fam = matched_family()
    coarse, fine_iso, flagship = fam["coarse"], fam["fine_iso"], fam["flagship"]

    # closed form: totals and actives
    assert expert_param_count(flagship, D) == expert_param_count(coarse, D)
    assert (active_expert_param_count(flagship, D)
            == active_expert_param_count(coarse, D)
            == active_expert_param_count(fine_iso, D)
            == 2 * D * fam["dense_hidden"])

    # torch counts agree with the closed form
    for spec in (coarse, fine_iso, flagship):
        layer = MoELayer(D, spec)
        n = layer.w_in.numel() + layer.w_out.numel()
        if layer.shared is not None:
            n += sum(p.numel() for p in layer.shared.parameters())
        assert n == expert_param_count(spec, D)

    # full models: identical except the router (d * N per layer)
    data = tiny_data()
    m_flag = CharLM(data.vocab_size, 4, moe_spec=flagship)
    m_coarse = CharLM(data.vocab_size, 4, moe_spec=coarse)
    router_delta = 2 * D * (flagship.n_experts - coarse.n_experts)  # 2 layers
    assert m_flag.n_params() - m_coarse.n_params() == router_delta


def test_plain_top2_fine_cannot_match_both_counts():
    """With uniform experts active/total == k/N, so plain 8-expert top-2 (1/4)
    can never match 2-expert top-1 (1/2) on both counts — the shared expert is
    what makes the exact match solvable."""
    fam = matched_family()
    assert active_total_ratio(fam["fine_iso"]) == pytest.approx(1 / 4)
    assert active_total_ratio(fam["coarse"]) == pytest.approx(1 / 2)
    # and indeed fine_iso, matched on active, pays 2x total:
    assert expert_param_count(fam["fine_iso"], D) == 2 * expert_param_count(fam["coarse"], D)
    # while the shared expert closes the algebra exactly:
    assert active_total_ratio(fam["flagship"]) == pytest.approx(1 / 2)


# ------------------------------------------------------- aux-free balancing

def test_bias_flips_selection_but_never_gate_weights():
    layer = small_layer(balance="auxfree", renormalize=True)
    x = torch.randn(3, 8)
    logits = layer.router(x)
    probs = F.softmax(logits, dim=-1)

    top_unbiased, gates_unbiased, _ = layer.route(x)

    # craft a bias that forces the WORST-ranked expert of token 0 into top-k
    worst = logits[0].argmin().item()
    layer.expert_bias.zero_()
    layer.expert_bias[worst] = 100.0
    top_biased, gates_biased, _ = layer.route(x)

    # selection changed: the worst expert is now selected for every token
    assert worst not in top_unbiased[0].tolist()
    assert all(worst in row for row in top_biased.tolist())

    # gates still come from the UNBIASED softmax, renormalized over selection
    expected = probs.gather(1, top_biased)
    expected = expected / expected.sum(-1, keepdim=True)
    assert torch.allclose(gates_biased, expected, atol=1e-6)
    # the biased score (logit + 100) never leaks into the gate values
    assert gates_biased.max() <= 1.0


def test_bias_that_keeps_selection_leaves_output_identical():
    layer = small_layer(balance="auxfree")
    x = torch.randn(2, 3, 8)
    out_ref = layer(x)
    top_ref = layer.last_top_idx.clone()
    with torch.no_grad():
        layer.expert_bias += 1e-4  # uniform tiny bias: cannot change any top-k
    out_biased = layer(x)
    assert torch.equal(layer.last_top_idx, top_ref)
    assert torch.allclose(out_biased, out_ref, atol=1e-7)


def test_bias_update_rule_sign_and_magnitude():
    gamma = 0.01
    layer = small_layer(balance="auxfree", gamma=gamma)
    layer.last_load = torch.tensor([0.5, 0.3, 0.1, 0.1])
    layer.expert_bias.zero_()
    layer.update_bias()
    # err = mean(0.25) - load -> signs [-1, -1, +1, +1], magnitude exactly gamma
    assert torch.allclose(layer.expert_bias,
                          torch.tensor([-gamma, -gamma, gamma, gamma]))
    # perfectly balanced load -> sign(0) == 0 -> no drift
    layer.expert_bias.zero_()
    layer.last_load = torch.full((4,), 0.25)
    layer.update_bias()
    assert torch.equal(layer.expert_bias, torch.zeros(4))
    # rule is inert unless balance == "auxfree"
    inert = small_layer(balance="aux")
    inert.last_load = torch.tensor([0.5, 0.3, 0.1, 0.1])
    inert.update_bias()
    assert torch.equal(inert.expert_bias, torch.zeros(4))


def test_bias_is_a_buffer_not_a_parameter():
    layer = small_layer(balance="auxfree")
    assert not layer.expert_bias.requires_grad
    assert "expert_bias" not in dict(layer.named_parameters())
    x = torch.randn(2, 3, 8)
    layer(x).sum().backward()          # grads flow fine with the bias in place
    assert layer.expert_bias.grad is None


# ----------------------------------------------------------- shared expert

def test_shared_expert_always_contributes_forward_and_grad():
    layer = small_layer(shared_hidden=6)
    x = torch.randn(2, 4, 8)
    flat = x.reshape(-1, 8)
    out = layer(x).reshape(-1, 8)

    # forward: output == routed reference + shared(x), token by token
    top_idx, gates, _ = layer.route(flat)
    routed = torch.zeros_like(flat)
    for m in range(flat.shape[0]):
        for slot in range(layer.k):
            e = top_idx[m, slot].item()
            routed[m] += gates[m, slot] * layer.run_expert(e, flat[m:m + 1])[0]
    assert torch.allclose(out, routed + layer.shared(flat), atol=1e-5)
    # the shared path moves EVERY token (not just routed ones)
    assert (layer.shared(flat).norm(dim=-1) > 0).all()

    # gradient reaches the shared expert from every token independently
    layer.zero_grad()
    layer(x).sum().backward()
    assert layer.shared.w_in.weight.grad is not None
    assert layer.shared.w_in.weight.grad.abs().sum() > 0

    # shared contribution is independent of the router: even with the router
    # zeroed (uniform routing), the output still decomposes as routed + shared
    with torch.no_grad():
        layer.router.weight.mul_(0.0)
    out2 = layer(x).reshape(-1, 8)
    top2, gates2, _ = layer.route(flat)
    routed2 = torch.zeros_like(flat)
    for m in range(flat.shape[0]):
        for slot in range(layer.k):
            e = top2[m, slot].item()
            routed2[m] += gates2[m, slot] * layer.run_expert(e, flat[m:m + 1])[0]
    assert torch.allclose(out2, routed2 + layer.shared(flat), atol=1e-5)


# ------------------------------------------------------------------- gating

def test_gate_renormalization_flag():
    x = torch.randn(5, 8)
    layer = small_layer(renormalize=True)
    _, gates, _ = layer.route(x)
    assert torch.allclose(gates.sum(-1), torch.ones(5), atol=1e-6)

    torch.manual_seed(0)
    layer_bug = small_layer(renormalize=False)  # same weights (same seed)
    _, gates_bug, probs = layer_bug.route(x)
    # un-renormalized top-k softmax gates strictly under-weight the mixture
    assert (gates_bug.sum(-1) < 1.0).all()
    top_idx, _, _ = layer_bug.route(x)
    assert torch.allclose(gates_bug, probs.gather(1, top_idx), atol=1e-7)


def test_top2_dispatch_matches_bruteforce_reference():
    layer = small_layer(n_experts=6, k=2, shared_hidden=7)
    x = torch.randn(3, 5, 8)
    flat = x.reshape(-1, 8)
    out = layer(x).reshape(-1, 8)
    top_idx, gates, _ = layer.route(flat)
    ref = layer.shared(flat).clone()
    for m in range(flat.shape[0]):
        for slot in range(layer.k):
            e = top_idx[m, slot].item()
            ref[m] += gates[m, slot] * layer.run_expert(e, flat[m:m + 1])[0]
    assert torch.allclose(out, ref, atol=1e-5)
    # load stats: fractions over routed slots, sum to 1
    assert layer.last_load.shape == (6,)
    assert layer.last_load.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------- aux loss

def test_aux_loss_uniform_and_collapsed_invariants():
    layer = small_layer(balance="aux", k=1)
    x = torch.randn(4, 8, 8)
    # uniform router -> P uniform -> aux == 1 regardless of tie-broken loads
    with torch.no_grad():
        layer.router.weight.zero_()
    layer(x)
    assert layer.last_aux.item() == pytest.approx(1.0, abs=1e-5)
    # collapsed router (all mass on expert 0) -> aux == N
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.weight[0] += 100.0 / x.shape[-1]
        # make logit 0 huge via input-independent trick: bias-free, so use input
    x_pos = torch.ones(4, 8, 8) * 3.0
    layer(x_pos)
    assert layer.last_aux.item() == pytest.approx(layer.n_experts, rel=1e-3)


# --------------------------------------------------------------------- MTP

def test_mtp_targets_shift_and_mask():
    data = tiny_data(block_size=4)
    gen = np.random.default_rng(0)
    x, y1, y2, dom = data.get_batch("train", 8, gen)
    assert x.shape == y1.shape == y2.shape == dom.shape == (8, 4)
    # y1 is x shifted by one; y2 is x shifted by two (within the window)
    assert torch.equal(y1[:, :-1], x[:, 1:])
    assert torch.equal(y2[:, :-2], x[:, 2:])
    assert torch.equal(y2[:, :-1], y1[:, 1:])
    # the final position has NO valid t+2 target inside the window
    assert (y2[:, -1] == IGNORE_INDEX).all()
    assert (y2[:, :-1] != IGNORE_INDEX).all()


def test_mtp_loss_ignores_masked_position():
    data = tiny_data(block_size=4)
    gen = np.random.default_rng(0)
    _, _, y2, _ = data.get_batch("train", 4, gen)
    V = data.vocab_size
    logits = torch.randn(4, 4, V, requires_grad=True)
    loss = F.cross_entropy(logits.reshape(-1, V), y2.reshape(-1),
                           ignore_index=IGNORE_INDEX)
    # perturbing the masked (last) position changes nothing
    with torch.no_grad():
        perturbed = logits.clone()
        perturbed[:, -1, :] += 100.0
    loss_p = F.cross_entropy(perturbed.reshape(-1, V), y2.reshape(-1),
                             ignore_index=IGNORE_INDEX)
    assert loss_p.item() == pytest.approx(loss.item(), abs=1e-6)
    # and no gradient reaches the masked position
    loss.backward()
    assert torch.allclose(logits.grad[:, -1, :], torch.zeros(4, V))
    assert logits.grad[:, :-1, :].abs().sum() > 0


def test_mtp_head_only_changes_training_not_lm_forward():
    data = tiny_data(block_size=4)
    torch.manual_seed(0)
    m_plain = CharLM(data.vocab_size, 4, dense_hidden=16, mtp=False)
    torch.manual_seed(0)
    m_mtp = CharLM(data.vocab_size, 4, dense_hidden=16, mtp=True)
    x = torch.randint(0, data.vocab_size, (2, 4))
    logits_plain, none = m_plain(x)
    logits_mtp, extra = m_mtp(x)
    assert none is None and extra is not None
    assert torch.allclose(logits_plain, logits_mtp, atol=1e-6)
    assert extra.shape == logits_mtp.shape


# -------------------------------------------------------------- determinism

def test_training_is_deterministic():
    text, tags = build_corpus(n_chars=6000, seed=0)
    data = CharData(text, tags, block_size=32)
    fam = matched_family(unit=4)
    vb = make_val_batches(data, n_batches=2, batch_size=4)
    runs = []
    for _ in range(2):
        torch.manual_seed(0)
        model = CharLM(data.vocab_size, 32, moe_spec=fam["flagship"])
        runs.append(train_model(model, data, vb, steps=8, batch_size=4,
                                eval_every=4, seed=0))
    assert runs[0]["train_loss"] == runs[1]["train_loss"]
    assert runs[0]["final"]["val_loss"] == runs[1]["final"]["val_loss"]
    assert np.array_equal(runs[0]["biases"][0], runs[1]["biases"][0])
    # and the aux-free controller actually moved the biases
    assert np.abs(runs[0]["biases"][0][-1]).sum() > 0


def test_corpus_has_all_three_domains_aligned():
    text, tags = build_corpus(n_chars=20000, seed=1)
    assert len(text) == len(tags)
    assert set(np.unique(tags)) == {0, 1, 2}
    # code blocks contain defs, number blocks contain '=' math lines
    assert "def " in text and " = " in text
