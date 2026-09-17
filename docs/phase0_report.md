# Phase 0 — Scope and Configuration Consistency

Status: **complete**. Full decision record: [`phase0_decision.md`](phase0_decision.md).
Verification: `test_phase0_config_consistency.py`.

## Purpose

Fix the research question and make the codebase consistent with it before any
experiment is run. Without this, later phases would measure artifacts of a
misconfigured default path rather than algorithm behaviour.

## Decisions taken

| # | Decision |
|---|---|
| 1 | **Primary question is algorithm selection**, not EA operator tuning. |
| 2 | `EXTENDED_CONFIG_SPACE` is the canonical config space — it includes the `algorithm` key. |
| 3 | EA operator tuning is retained as a **secondary** experiment, pinned to `algorithm=["ea"]`. |
| 4 | The LLM designer proposes across the full algorithm set, not operators alone. |
| 5 | Every entry point must pass an explicit `algorithm` key. |

## Key finding

The default path was **broken** before Phase 0. Several entry points built
configurations without an `algorithm` key, and `EA_CONFIG_SPACE` was the default
in `config_space.py` while `EXTENDED_CONFIG_SPACE` was the intended space. The
effect was that "default" runs silently exercised a single hardcoded algorithm
rather than the selection space.

Fixed by: defaulting all four `config_space.py` entry points to
`EXTENDED_CONFIG_SPACE`, and making `build_algorithm` raise `ValueError` on a
missing `algorithm` key instead of silently defaulting to PSO.

## Budget protocol fixed here

Binding on every later phase:

| Parameter | Value |
|---|---|
| `population_size` | 50 |
| `max_fe` | 5000 (equal **evaluation** budget) |
| `ea` equivalent | `max_generations=100` |

Equal evaluations, not equal generations: population methods spend `pop_size`
evaluations per generation while single-solution methods such as SA spend
roughly one, so equal generations would not be a fair comparison.

## Still open at close of Phase 0

Item 6 (deferred) and the items listed under §7c of the decision record.

## Reproducing

```bash
python test_phase0_config_consistency.py
```
