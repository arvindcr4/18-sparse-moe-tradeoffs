"""Training / evaluation loop with load, bias and routing instrumentation."""

import numpy as np
import torch
import torch.nn.functional as F

from .corpus import IGNORE_INDEX, CharData
from .model import CharLM


def make_val_batches(data: CharData, n_batches: int = 8, batch_size: int = 24,
                     seed: int = 123):
    """A FIXED set of validation batches so every run is scored identically."""
    gen = np.random.default_rng(seed)
    return [data.get_batch("val", batch_size, gen) for _ in range(n_batches)]


@torch.no_grad()
def evaluate(model: CharLM, val_batches) -> dict:
    """Mean next-token CE, and (if the model has an MTP head) t+2 CE and top-1 acc."""
    model.eval()
    ce1, ce2, correct, valid = 0.0, 0.0, 0, 0
    for x, y1, y2, _ in val_batches:
        logits, mtp_logits = model(x)
        ce1 += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               y1.reshape(-1)).item()
        if mtp_logits is not None:
            flat = mtp_logits.reshape(-1, mtp_logits.size(-1))
            tgt = y2.reshape(-1)
            ce2 += F.cross_entropy(flat, tgt, ignore_index=IGNORE_INDEX).item()
            mask = tgt != IGNORE_INDEX
            correct += (flat[mask].argmax(-1) == tgt[mask]).sum().item()
            valid += int(mask.sum())
    model.train()
    out = {"val_loss": ce1 / len(val_batches)}
    if valid:
        out["mtp_loss"] = ce2 / len(val_batches)
        out["mtp_acc"] = correct / valid
    return out


def train_model(model: CharLM, data: CharData, val_batches, steps: int = 800,
                batch_size: int = 24, lr: float = 3e-3, mtp_coef: float = 0.5,
                eval_every: int = 40, seed: int = 0) -> dict:
    """Train and record: train/val loss curves, per-layer expert loads and biases.

    The total loss is ``CE(next) + aux_coef * sum(aux) + mtp_coef * CE(t+2)``;
    aux-free bias updates run after every optimizer step.
    """
    torch.manual_seed(seed)
    gen = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    moe = model.moe_layers
    hist = {
        "train_loss": [],
        "val_steps": [], "val_loss": [],
        "loads": [[] for _ in moe],   # per layer: list of (N,) arrays
        "biases": [[] for _ in moe],  # per layer: list of (N,) arrays
    }
    model.train()
    for step in range(steps):
        x, y1, y2, _ = data.get_batch("train", batch_size, gen)
        logits, mtp_logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y1.reshape(-1))
        for layer in moe:
            if layer.last_aux is not None:
                loss = loss + layer.spec.aux_coef * layer.last_aux
        if mtp_logits is not None:
            loss = loss + mtp_coef * F.cross_entropy(
                mtp_logits.reshape(-1, mtp_logits.size(-1)), y2.reshape(-1),
                ignore_index=IGNORE_INDEX)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.update_biases()

        hist["train_loss"].append(float(loss.item()))
        for i, layer in enumerate(moe):
            hist["loads"][i].append(layer.last_load.numpy().copy())
            hist["biases"][i].append(layer.expert_bias.numpy().copy())
        if step % eval_every == 0 or step == steps - 1:
            hist["val_steps"].append(step)
            hist["val_loss"].append(evaluate(model, val_batches)["val_loss"])

    hist["final"] = evaluate(model, val_batches)
    hist["loads"] = [np.stack(l) for l in hist["loads"]]
    hist["biases"] = [np.stack(b) for b in hist["biases"]]
    return hist


@torch.no_grad()
def domain_routing_counts(model: CharLM, data: CharData, n_batches: int = 12,
                          batch_size: int = 24, seed: int = 7) -> np.ndarray:
    """Routing counts per (layer, domain, expert) on validation data.

    Returns an array of shape (n_moe_layers, 3, N): how many top-k slots each
    expert received from tokens of each domain.
    """
    model.eval()
    moe = model.moe_layers
    counts = np.zeros((len(moe), 3, moe[0].n_experts))
    gen = np.random.default_rng(seed)
    for _ in range(n_batches):
        x, _, _, dom = data.get_batch("val", batch_size, gen)
        model(x)
        dom_flat = dom.reshape(-1)
        for li, layer in enumerate(moe):
            top = layer.last_top_idx  # (M, k)
            for d in range(3):
                sel = top[dom_flat == d].reshape(-1)
                counts[li, d] += np.bincount(sel.numpy(),
                                             minlength=layer.n_experts)
    model.train()
    return counts
