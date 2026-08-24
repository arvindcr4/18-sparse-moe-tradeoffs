# Modern Sparse-Model Trade-Offs

> Project 18 of 34 — [Step-By-Step LLM Engineering Projects (2026 Edition)](https://x.com/TheAhmadOsman/status/2058745340895870985) by [@TheAhmadOsman](https://x.com/TheAhmadOsman)
>
> **Part X: Mixture Of Experts**

## Goal

Reproduce, at toy scale, the trade-offs behind modern sparse systems: fine-grained and shared experts, auxiliary-loss-free load balancing, and multi-token prediction — the DeepSeek-V3 / Llama 4 / Qwen3 / Kimi K2 design space.

## The project loop

1. **Build it.** Implement the core idea yourself before using the library version.
2. **Plot it.** Loss curves, latency curves, memory curves, attention heatmaps, routing histograms, entropy plots, or failure galleries.
3. **Break it.** Ablate the thing you just built and watch what changes.
4. **Explain it.** A short technical note: what you expected, what happened, what surprised you, what you'd try next.

## Artifacts to ship

- [ ] **Implementation** — clean code with tests
- [ ] **Notebook** — one reproducible experiment
- [ ] **Plots** — at least three charts showing behavior
- [ ] **Failure gallery** — examples where the system breaks
- [ ] **Write-up** — what I learned and what changed my mind

---

*Status: scaffolded — build in progress.*
