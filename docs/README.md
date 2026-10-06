# Phase reports

One report per phase, written when all steps of that phase complete without
error. Every figure in these reports is literal output from code run against
the files in `results/`. No recalled or hand-computed numbers.

Updated 2026-10-01 for the rebuilt data: a 72-row evaluation matrix and a
720-row training matrix, both covering dims 5, 10 and 20. Reports for Phases
2, 2c, 3 and 3b were rewritten. Phases 0, 1 and 1b use fixed setups that the
rebuild does not touch, and were re-checked.

| Phase | Report | Headline |
|---|---|---|
| 0 | [phase0_report.md](phase0_report.md) · [decision record](phase0_decision.md) | Scope fixed on algorithm selection; broken default path found and fixed |
| 1 | [phase1_report.md](phase1_report.md) | PSO beats GA on F1 Sphere, `U=0.0`, `p=3.02e-11`, `r=1.0` |
| 1b | [phase1b_report.md](phase1b_report.md) | 21 screened; `CMA_ES`, `AOA`, `DE` removed as non-optimising |
| 2 | [phase2_report.md](phase2_report.md) | EO (4.722) and MPA (4.847) jointly strongest, indistinguishable (p=0.5894); best algorithm depends on dimension; SBS matches the oracle on 11 of 72 rows |
| 2c | [phase2c_report.md](phase2c_report.md) | 720-row training matrix, all 9 features live; no selector beats SBS on it (RF-regression −22.9%, p=0.356) |
| 3 | [phase3_report.md](phase3_report.md) | Rank-target selector ties SBS (−0.3%, p=0.677); no cell of the 2×2 ablation beats SBS |
| 3b | [phase3b_report.md](phase3b_report.md) | Pre-registered test on independent data: +2.1%, p=0.526, so the Phase 3 result **does not replicate** |

The full plan, including Phases 4 and 5 and the thesis write-up, is in
[project_plan.md](project_plan.md).

## Authoritative sources

- **`results/phase2_matrix.json`**: 72 rows, seed 100, `n_runs=15`. The source
  for any SBS/VBS figure quoted in the thesis, and the independent evaluation
  set for Phase 3b.
- **`results/phase2c_matrix.json`**: 720 rows, seeds 42–51, `n_runs=5`.
  **Selector training only**, marked non-authoritative in its own metadata.

The two matrices share no seed (100 against 42–51), so the Phase 3b split is
independent. `phase3b_holdout_confirm.py` refuses to run if they ever overlap.

## Main finding

The algorithms differ strongly (Friedman p ≈ 6.27e-139 over 72 rows), and the
landscape features identify the best algorithm almost perfectly when a function
has been seen (in-sample ceiling +97.7%). But a selector trained on other
functions does not beat always running the single best solver on an unseen
function. That holds in every analysis of the rebuilt data, including the
pre-registered confirmation test.

## Open caveats carried forward

1. **Training labels are noisy.** Training cells are medians of 5 runs. At 5
   runs, about 1 row in 6 has a best-algorithm label that is off by more than
   1% (see the Phase 2c report).
2. **24 functions bound every test.** Rows within a function are related, so
   each significance test pairs at most 24 function means. The tests cannot
   rule out a small benefit; the measured one in Phase 3b is 0.08 of a rank.
3. **One evaluation seed.** The evaluation matrix has one landscape realisation
   (seed 100) per function and dimension.
4. **F5 LinearSlope barely discriminates.** Up to 18 of 22 algorithms tie at its
   exact optimum.

## Note on `results/`

`results/` is tracked in git, apart from the `.jsonl` crash journal and other
temporary files. `results/MANIFEST.md` lists what each file is, its rebuild
cost, and how the backup folder is kept in sync.
