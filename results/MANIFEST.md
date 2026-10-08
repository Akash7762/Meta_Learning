# results/ — what is needed and what is not

Updated 2026-10-08. `results/` is **tracked in git** (since commit `b102067`),
except for transient files listed under LOCAL ONLY. Everything marked KEEP is
either expensive to rebuild or is the current evidence for a thesis claim.

Regenerate anything from the scripts; see `docs/README.md` for the run order.

---

## ✅ KEEP — current pipeline output

| File | What it is | Cost to rebuild |
|---|---|---|
| `phase2_matrix.json` | **Authoritative evaluation matrix.** 72 rows (24 functions × dims 5/10/20), seed 100, n_runs=15, 1584 cells, 2.1 MB. Headline SBS/VBS figures come from here. | **8.45 h** |
| `phase2_matrix.csv` | Human-readable export of the above | seconds |
| `phase2c_matrix.json` | **Selector training matrix.** 720 rows (24 functions × dims 5/10/20 × seeds 42–51), n_runs=5, 15,840 cells, 13 MB. | **31.25 h** — the most expensive artifact in the project |
| `phase2_lopo.json` / `.txt` | LOPO on the evaluation matrix, function-mean statistics. RF-regression +30.8%, p=0.053 (not significant). | ~3 min |
| `phase2c_lopo.json` / `.txt` | LOPO on the training matrix. No selector beats SBS; RF-regression −22.9%, p=0.356. | minutes |
| `phase3_selector.json` / `.txt` | Step 11, rank-target selector. −0.3% gap closed, p=0.677. | minutes |
| `phase3_ablation.json` | Step 12, target × architecture 2×2. No design beats SBS; best cell +0.8%. | ~23 min |
| `phase3b_holdout_confirm.json` / `.txt` | Step 13, the pre-registered test: train on the 720-row matrix, evaluate on the independent 72-row matrix. +2.1% gap closed, p=0.526: the Phase 3 result does not replicate. | ~15 min |
| `phase2c_label_noise.png` / `.txt` / `_rows.csv` / `_functions.csv` | How noisy the training labels are: each row's best algorithm re-picked from 1,000 bootstrap resamples of its 5 stored runs (`phase2c_label_noise.py`). Mean label agreement 66.5%; 30.1% of rows unstable; 16.9% of re-picks more than 1% worse than the best. | seconds |
| `phase1b_screening.json` / `.txt` | Algorithm health screen; evidence for removing CMA_ES, AOA and DE. | minutes |
| `phase4a_dev.json` / `.txt` | Phase 4a development replay on the training matrix: 4 `EA_Designer` variants, 1 replay per row. Chose v3 (meta-population 10, surrogate on), mean score −0.0116. | 1496.4 s on 8 workers |
| `phase4b_confirm.json` / `.txt` | Phase 4b, the pre-registered test on the evaluation matrix, 30 replays per row: `EA_Designer` does not beat SBS at an equal budget of 15 runs (−46.0% gap closed, p=0.252). | 4232.7 s on 8 workers |

Each full rebuild of `phase2c_matrix.json` adds ~13 MB to the repository:
git stores a whole new blob per change, not a delta.

## ⏳ EXPECTED

Nothing. Steps 1–15 have all been run.

## 💾 LOCAL ONLY — gitignored

| File | Why |
|---|---|
| `phase2c_matrix.jsonl` | Append-only crash journal (9.7 MB). Only used to resume an interrupted build; the build is complete, so it is redundant. Safe to keep or delete. |
| `*.tmp`, `_tmp_*`, `*_dryrun.*` | Scratch and atomic-write temporaries. |

## 🗑️ GONE — the 89 PNGs

Deleted deliberately on 2026-09-23. They came from `experiment_runner.py` and
`user_model_runner.py`, which sit **outside** the Phase 0–3b pipeline, so
running steps 1–13 will not recreate them. If a thesis figure cites one,
re-run those two scripts. Nothing in Phase 0–3b depends on them.

---

## Backup — `~/meta_learning_backup/`

**In sync as of 2026-10-08:** every file in this folder is mirrored there,
verified by md5 (23 of 23 matching). The `.jsonl` crash journal is not copied.
The folder also keeps copies of `docs/phase3_preregistration.md` and
`docs/phase4_preregistration.md`.

**It also holds a file that exists nowhere else:**
`phase2c_matrix.96row_n3_legacy.json`, the superseded **96-row, dim-10,
n_runs=3** build (md5 `9a158b9363d249935bb659580998ad63`, identical to the
original of 2026-09-16). It is the data behind the original Phase 3 result,
and the training half of the original Phase 3b result, both superseded and
recorded in `docs/phase3_preregistration.md`. The seed-42, dim-10 evaluation
matrix that the original Phase 3b run was scored on no longer exists anywhere: it was overwritten in the backup on
2026-09-23. The legacy file predates `results/` being tracked, so it was never
committed. It can be regenerated from the code at commit `90c2e9f` (~2.4 h),
but that build did not record library versions, so bit-identical
regeneration is not guaranteed. Keep it if the thesis discusses the original
run.

The legacy copy has its own name, so a plain re-sync cannot overwrite it.
Re-sync after any long build:

```bash
cp -f results/*.json results/*.txt results/*.csv results/*.md results/*.png ~/meta_learning_backup/
```

```bash
cp -f docs/phase3_preregistration.md docs/phase4_preregistration.md ~/meta_learning_backup/
```

Do not rename or delete `phase2c_matrix.96row_n3_legacy.json`.
