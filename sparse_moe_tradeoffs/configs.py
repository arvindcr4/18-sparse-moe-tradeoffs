"""Closed-form parameter-matching algebra for the fine-vs-coarse MoE family.

All experts are bias-free 2-layer MLPs, so an expert with hidden width ``h``
on a trunk of width ``d`` has exactly ``2 * d * h`` parameters.  For a uniform
MoE with ``N`` experts of width ``h`` routed top-``k``:

    total  expert params = N * 2dh
    active expert params = k * 2dh          (per token)

so the active/total ratio is ``k / N`` regardless of ``h``.  A plain
fine-grained config (N=8, top-2, ratio 1/4) therefore can NEVER match a coarse
config (N=2, top-1, ratio 1/2) on both total AND active parameters — the
matching only becomes solvable once a DeepSeek-style always-active shared
expert is added.  With routed width ``u``, shared width ``s`` and coarse width
``h_c`` the two conditions

    total :  8u + s  = 2 h_c
    active:  2u + s  =   h_c

have the exact solution ``s = 4u``, ``h_c = 6u``.  That algebra fixes the
whole family used in the experiments (``u = 32`` by default, d = 96):

    dense      : one FFN, hidden 6u                    (iso-active reference)
    coarse     : N=2,  top-1, hidden 6u                (total 12u·2d, active 6u·2d)
    fine_iso   : N=8,  top-2, hidden 3u, no shared     (active 6u·2d matched,
                                                        total 24u·2d — 2x, unavoidable)
    flagship   : N=8,  top-2, hidden u, shared 4u      (total 12u·2d AND active
                                                        6u·2d — both match coarse)
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MoESpec:
    """Configuration of one MoE feed-forward layer."""

    n_experts: int
    k: int
    expert_hidden: int
    shared_hidden: int = 0            # 0 = no shared expert
    balance: str = "auxfree"          # "aux" | "auxfree" | "none"
    gamma: float = 0.01               # aux-free bias update rate
    aux_coef: float = 0.0             # weight of the auxiliary loss (balance == "aux")
    renormalize: bool = True          # renormalize the top-k gate weights to sum to 1


def expert_param_count(spec: MoESpec, d: int) -> int:
    """Closed-form TOTAL parameter count of the expert MLPs (routed + shared).

    The router (``d * n_experts`` weights) is excluded: it is the only part of
    the layer whose size depends on N alone and it is negligible (< 1k params
    here); tests account for it separately.
    """
    return 2 * d * (spec.n_experts * spec.expert_hidden + spec.shared_hidden)


def active_expert_param_count(spec: MoESpec, d: int) -> int:
    """Closed-form ACTIVE (per-token) parameter count: top-k routed + shared."""
    return 2 * d * (spec.k * spec.expert_hidden + spec.shared_hidden)


def active_total_ratio(spec: MoESpec) -> float:
    """Fraction of expert parameters active per token."""
    return active_expert_param_count(spec, 1) / expert_param_count(spec, 1)


def matched_family(unit: int = 32) -> dict:
    """Build the exactly-matched config family described in the module docstring.

    Returns a dict with keys ``dense_hidden``, ``coarse``, ``fine_iso`` and
    ``flagship``.  Raises AssertionError if the algebra is violated.
    """
    u = unit
    # Top-1 uses the Switch convention (gate = raw router prob): renormalizing
    # a single gate would pin it to 1.0 and cut the router off from gradients.
    coarse = MoESpec(n_experts=2, k=1, expert_hidden=6 * u, renormalize=False)
    fine_iso = MoESpec(n_experts=8, k=2, expert_hidden=3 * u)
    flagship = MoESpec(n_experts=8, k=2, expert_hidden=u, shared_hidden=4 * u)

    d = 1  # counts scale linearly in d; verify with d=1
    assert expert_param_count(flagship, d) == expert_param_count(coarse, d)
    assert active_expert_param_count(flagship, d) == active_expert_param_count(coarse, d)
    assert active_expert_param_count(fine_iso, d) == active_expert_param_count(coarse, d)

    return {
        "dense_hidden": 6 * u,
        "coarse": coarse,
        "fine_iso": fine_iso,
        "flagship": flagship,
    }


def variant(spec: MoESpec, **overrides) -> MoESpec:
    """Return a copy of ``spec`` with fields overridden (for ablations)."""
    return replace(spec, **overrides)
