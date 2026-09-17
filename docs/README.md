# Phase Reports

One report per phase, written when all steps of that phase complete without
error. Every figure in these reports is literal output from code run against
cached data — no recalled or hand-computed numbers.

| Phase | Report | Headline |
|---|---|---|
| 0 | [phase0_report.md](phase0_report.md) · [decision record](phase0_decision.md) | Scope fixed on algorithm selection; broken default path found and fixed |
| 1 | [phase1_report.md](phase1_report.md) | PSO beats GA on F1 Sphere, `U=0.0`, `p=3.02e-11`, `r=1.0` |
| 1b | [phase1b_report.md](phase1b_report.md) | 21 screened; `CMA_ES`, `AOA`, `DE` removed as non-optimising |
| 2 | [phase2_report.md](phase2_report.md) | **MPA is SBS** (mean rank 4.500); no selector beats it; headroom small |
| 2c | [phase2c_report.md](phase2c_report.md) | 96 rows; negative result replicates; `n_runs=3` deviation |
| 3 | [phase3_report.md](phase3_report.md) | Rank target → **+38.2%, p=0.0149**; target dominates architecture |
| 3b | [phase3b_report.md](phase3b_report.md) | Pre-registered confirm: **+52.8%, p=0.0072** |

## Authoritative sources

- **`results/phase2_matrix.json`** (`n_runs=15`, 24 rows) — authoritative for any
  SBS/VBS figure quoted in the thesis.
- **`results/phase2c_matrix.json`** (`n_runs=3`, 96 rows) — **selector training
  only**, marked non-authoritative in its own metadata.

## Open caveats carried forward

1. **Post-hoc target selection is unresolved.** The rank target was chosen after
   seeing it win an 8-cell ablation; Phase 3b could not test it, because no
   problem configuration exists outside Phase 2c's training set.
2. **`n_runs=3`** on all selector training data, against a spec of 15.
3. **dim=10 only.** The `dimension` feature is constant and carries no signal;
   dims 5 and 20 are a planned expansion. Code keeps `dim` a live variable, so
   adding them is a data change, not a code change.
4. **Small headroom.** MPA already matches the oracle on 10 of 24 problems.

## Note on `results/`

`results/` is gitignored. Matrices and output files are **not** in version
control and must be regenerated, or restored from a manual backup.
