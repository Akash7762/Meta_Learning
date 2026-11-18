# Project documentation (v2)

v2 is a fresh start of the research, begun 2026-11-18. It keeps the fixed
framework code (`aad/`) but none of the round-one phases, data or reports.
Every number in a v2 report must come from v2 result files.

## Folder layout

```
docs/
├── README.md            this index
├── planning/phase<N>/   everything written BEFORE a phase runs: plans,
│                        decisions, pre-registrations (never edited once committed)
└── reports/             one report per completed phase: phase<N>_report.md
results/
├── MANIFEST.md          what each result file is and whether it is current
└── phase<N>/            one folder per phase
```

A phase gets its `planning/phase<N>/` folder when its planning starts, and its
report when all its steps finish without error.

## Phases

| Phase | Plan | Report | Status |
|---|---|---|---|
| 0 | — | — | not started: decide the rules |

## Round one (v1), kept for reference only

Nothing from v1 is deleted; it is not part of v2's evidence.

| Where | What |
|---|---|
| git tag `v1-final` (commit `bf2c22d`) | the complete v1 project: code, docs, tracked results |
| folder `../Meta_Learning_v1` | the same tag checked out beside this one (a git worktree), plus the files git does not track: the Phase 2c crash journal, the Phase 5 call journal, the `experiment_runner.py` outputs |
| `~/meta_learning_backup/v1-extras/` | a second copy of those untracked files |
| `../../Meta_Learning-Original file/Meta_Learning` | the project's very first state (matches commit `bd71fa0`), outside git |

v1 headline, for orientation only: neither a feature-based selector
(+2.1%, p=0.526) nor the meta-learner `EA_Designer` (−46.0%, p=0.252) beat
always running the single best solver at an equal budget. Details are in
`docs/reports/` in the `v1-final` tag.
