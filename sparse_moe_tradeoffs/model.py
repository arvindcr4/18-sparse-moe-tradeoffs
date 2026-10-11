"""Tiny char transformer trunk with pluggable dense / sparse-MoE FFN layers.

Implements the modern sparse-MoE design space at toy scale:
  * fine-grained routed experts (top-k, token choice),
  * an always-active DeepSeek-style shared expert,
  * auxiliary-loss load balancing (Switch/GShard style),
  * auxiliary-loss-FREE balancing: a per-expert bias added to the router
    logits for TOP-K SELECTION ONLY (never the gate weights), updated by a
    sign rule ``b_i += gamma * sign(mean_load - load_i)``,
  * an optional multi-token-prediction (t+2) head on the trunk.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import MoESpec


class Expert(nn.Module):
    """Bias-free 2-layer GELU MLP: exactly ``2 * d * hidden`` parameters."""

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.w_in = nn.Linear(d, hidden, bias=False)
        self.w_out = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.gelu(self.w_in(x)))


class MoELayer(nn.Module):
    """Token-choice top-k sparse MoE FFN with optional shared expert.

    Gate weights are ALWAYS computed from the unbiased router logits
    (full softmax, then gathered at the selected experts).  The aux-free
    balancing bias enters only the argument of ``topk``.

    The routed experts are stored as two fused weight matrices (rows/columns
    grouped per expert) and evaluated as one big matmul with zero gates for
    unselected experts — mathematically identical to sparse per-expert
    dispatch (verified against a per-expert reference in the tests) but far
    faster on CPU at this scale.  ``2 * d * hidden`` params per expert exactly.
    """

    def __init__(self, d: int, spec: MoESpec):
        super().__init__()
        assert spec.balance in ("aux", "auxfree", "none")
        self.spec = spec
        self.n_experts, self.k = spec.n_experts, spec.k
        self.router = nn.Linear(d, spec.n_experts, bias=False)
        H = spec.n_experts * spec.expert_hidden
        self.w_in = nn.Parameter(torch.randn(H, d) * 0.02)   # rows grouped per expert
        self.w_out = nn.Parameter(torch.randn(d, H) * 0.02)
        self.shared = Expert(d, spec.shared_hidden) if spec.shared_hidden > 0 else None
        self.register_buffer("expert_bias", torch.zeros(spec.n_experts))
        # per-forward stats
        self.last_load = None      # (N,) fraction of routed slots per expert
        self.last_aux = None       # scalar aux loss tensor (balance == "aux")
        self.last_top_idx = None   # (B*T, k) selected expert ids

    def run_expert(self, e: int, x: torch.Tensor) -> torch.Tensor:
        """Evaluate routed expert ``e`` alone (reference path for tests)."""
        h = self.spec.expert_hidden
        sl = slice(e * h, (e + 1) * h)
        return F.gelu(x @ self.w_in[sl].T) @ self.w_out[:, sl].T

    def route(self, flat: torch.Tensor):
        """Router pass: returns ``(top_idx, gates)`` for flat tokens (M, d).

        ``top_idx`` comes from the BIASED logits (selection only); ``gates``
        come from the softmax of the ORIGINAL logits, optionally renormalized
        over the selected k to sum to 1.
        """
        logits = self.router(flat)                       # (M, N)
        probs = F.softmax(logits, dim=-1)                # unbiased: source of gates
        sel_scores = logits + self.expert_bias           # bias: SELECTION ONLY
        _, top_idx = torch.topk(sel_scores, self.k, dim=-1)
        gates = probs.gather(1, top_idx)                 # (M, k) from unbiased probs
        if self.spec.renormalize:
            gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return top_idx, gates, probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        M = B * T
        flat = x.reshape(-1, D)
        top_idx, gates, probs = self.route(flat)

        gate_full = torch.zeros(M, self.n_experts, dtype=flat.dtype,
                                device=flat.device)
        gate_full.scatter_(1, top_idx, gates)
        acts = F.gelu(flat @ self.w_in.T).view(M, self.n_experts, -1)
        weighted = (acts * gate_full.unsqueeze(-1)).view(M, -1)
        out = weighted @ self.w_out.T
        if self.shared is not None:
            out = out + self.shared(flat)

        with torch.no_grad():
            counts = torch.bincount(top_idx.reshape(-1),
                                    minlength=self.n_experts).float()
            self.last_load = counts / counts.sum()
        if self.spec.balance == "aux":
            # GShard/Switch style: N * sum_e f_e * P_e  (== 1 when uniform)
            self.last_aux = self.n_experts * (self.last_load * probs.mean(0)).sum()
        else:
            self.last_aux = None
        self.last_top_idx = top_idx.detach()
        return out.reshape(B, T, D)

    @torch.no_grad()
    def update_bias(self):
        """Aux-free balancing step: ``b_i += gamma * sign(mean_load - load_i)``.

        Underloaded experts (load below the mean) get their selection bias
        raised, overloaded ones lowered.  No gradient flows through the bias.
        """
        if self.spec.balance != "auxfree" or self.last_load is None:
            return
        err = self.last_load.mean() - self.last_load
        self.expert_bias += self.spec.gamma * torch.sign(err)


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention (bias-free projections)."""

    def __init__(self, d: int, n_head: int):
        super().__init__()
        assert d % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.n_head, D // self.n_head).transpose(1, 2)
                   for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    """Pre-norm transformer block with a pluggable FFN module."""

    def __init__(self, d: int, n_head: int, ffn: nn.Module):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_head)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = ffn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class CharLM(nn.Module):
    """2-layer char LM; FFN per block is dense or MoE; optional t+2 MTP head."""

    def __init__(self, vocab_size: int, block_size: int, d: int = 96,
                 n_layer: int = 2, n_head: int = 4,
                 moe_spec: MoESpec | None = None, dense_hidden: int = 192,
                 mtp: bool = False):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(block_size, d)

        def make_ffn():
            if moe_spec is None:
                return Expert(d, dense_hidden)
            return MoELayer(d, moe_spec)

        self.blocks = nn.ModuleList(Block(d, n_head, make_ffn())
                                    for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.apply(self._init)
        # Created AFTER trunk init so that a model with the MTP head has a
        # bit-identical trunk to one without it (clean A/B for the experiment).
        if mtp:
            self.mtp_norm = nn.LayerNorm(d)
            self.mtp_head = nn.Linear(d, vocab_size, bias=False)
            self._init(self.mtp_head)
        else:
            self.mtp_norm = self.mtp_head = None

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor):
        """Returns ``(logits, mtp_logits)``; ``mtp_logits`` is None without MTP."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        h = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        logits = self.head(h)
        mtp_logits = self.mtp_head(self.mtp_norm(h)) if self.mtp_head is not None else None
        return logits, mtp_logits

    @property
    def moe_layers(self) -> list:
        return [blk.ffn for blk in self.blocks if isinstance(blk.ffn, MoELayer)]

    def update_biases(self):
        """Apply the aux-free bias update on every MoE layer (post-step hook)."""
        for layer in self.moe_layers:
            layer.update_bias()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
