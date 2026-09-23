# results/ — what is needed and what is not

Written 2026-09-23. `results/` is **gitignored**: nothing here is in version
control. Everything marked KEEP is either expensive to rebuild or is the
current evidence for a thesis claim.

Regenerate anything from the scripts; see `docs/README.md` for the run order.

---

## ✅ KEEP — current pipeline output

| File | Why | Cost to rebuild |
|---|---|---|
| `phase2_matrix.json` | **The authoritative matrix.** 72 rows (24 functions × dims 5/10/20), seed 100, n_runs=15, 1584 cells. Every Phase 2 and Phase 3b figure derives from it. | **8.45 h** |
| `phase2_matrix.csv` | Human-readable export of the above | seconds |
| `phase2_lopo.json` / `.txt` | LOPO selector evaluation. Contains the corrected statistics — function-mean aggregation, not the inflated per-row test. | ~12 min |
| `phase1b_screening.json` / `.txt` | Algorithm health screen. The evidence for removing CMA_ES, AOA and DE from the registry. | ~3 min |

## ⏳ EXPECTED — not built yet

Produced by steps 8–13 of the run order.

| File | From |
|---|---|
| `phase2c_matrix.json` / `.jsonl` | `phase2c_expand_matrix.py` (**~30 h** — the most expensive artifact in the project) |
| `phase2c_lopo.json` / `.txt` | `phase2_lopo.py --matrix results/phase2c_matrix.json` |
| `phase3_selector.json` / `.txt` | `phase3_selector_validation.py` |
| `phase3_ablation.json` | `phase3_ablation.py` (optional) |
| `phase3b_holdout_confirm.json` / `.txt` | `phase3b_holdout_confirm.py` |

## 🗑️ GONE — the 89 PNGs

As of 2026-09-23 `results/` holds only the data files above. The ~89 plots
(`fitness_*.png`, `meta_*.png`, `cmp_*.png`, `fit_*.png`, `pareto_*.png`) are
no longer present.

They came from `experiment_runner.py` and `user_model_runner.py`, which sit
**outside** the Phase 0-3b pipeline, so running steps 1-13 will not recreate
them. If a thesis figure cites one, re-run those two scripts.

Nothing in Phase 0-3b depends on them.

---

## Backup

`~/meta_learning_backup/` mirrors every file in this folder (verified by md5
on 2026-09-23). It also keeps `phase2c_matrix.json` from the superseded 96-row
n_runs=3 build, and a copy of `phase3_preregistration.md` (the live one is
version-controlled at `docs/phase3_preregistration.md`).

results/ is now TRACKED IN GIT, so the backup is a second line of defence
rather than the only one. Re-sync after any long build:

```bash
cp -f results/*.json results/*.txt results/*.csv results/*.md ~/meta_learning_backup/
```
