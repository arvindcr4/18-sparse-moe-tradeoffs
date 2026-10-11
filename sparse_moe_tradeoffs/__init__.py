"""Modern sparse-MoE trade-offs at toy scale (DeepSeek-V3 / Mixtral design space)."""

from .configs import (MoESpec, active_expert_param_count, expert_param_count,
                      matched_family, variant)
from .corpus import IGNORE_INDEX, CharData, build_corpus
from .model import CharLM, Expert, MoELayer
from .train import (domain_routing_counts, evaluate, make_val_batches,
                    train_model)

__all__ = [
    "MoESpec", "matched_family", "variant",
    "expert_param_count", "active_expert_param_count",
    "CharData", "build_corpus", "IGNORE_INDEX",
    "CharLM", "MoELayer", "Expert",
    "train_model", "evaluate", "make_val_batches", "domain_routing_counts",
]
