# Project context for Claude

This repo is a Kaggle competition workspace for `playground-series-s6e5` — predicting `PitNextLap` (binary classification). Modeling work happens in `training.ipynb`; EDA in `eda.ipynb`. The reference setup ("baseline") and CV scheme are specified in `README.md` under **Baseline** — read that section before suggesting any modeling change so you know what every trial is being compared against.

## Trial logging workflow

Every modeling change that produces a CV score is a **trial** and must be recorded in `trials.csv`. The baseline itself is *not* a trial — it's the reference. A trial is anything that deviates from the baseline (new feature, removed feature, different algorithm, hyperparam change, different CV scheme, ensembling, etc.).

The workflow for a trial — follow this order, do not skip steps:

1. **Decide what's changing.** Be explicit with the user about which baseline knob is being turned (feature set, algorithm, hyperparams, CV) and why. One change per trial is preferred so the score delta is attributable; if multiple changes are bundled, call that out in `notes`.
2. **Make the code change** in `training.ipynb` (or wherever applicable). Don't touch unrelated cells.
3. **Run the full CV cell** end-to-end. Capture the OOF AUC printed at the bottom (`OOF AUC : ...`). If the user submitted to Kaggle, capture the public LB score too.
4. **Commit the code that produced the score.** The trial row is meaningless without a reproducible commit. Stage `training.ipynb` (and any other touched files) and commit with a short message describing the change. Do NOT commit `train.csv`, `test.csv`, `sample_submission.csv`, or `submission.csv` — they're gitignored / oversized. Get the short SHA with `git rev-parse --short HEAD`.
5. **Append a row to `trials.csv`** with the columns below. Increment `trial` by 1 from the last row. Use today's date in `YYYY-MM-DD`. The commit SHA goes in `commit`.
6. **Commit `trials.csv`** as a separate commit (message like `Log trial N: <change>`). Keeping the trial-log commit separate from the code commit means the `commit` column in `trials.csv` keeps pointing at the exact code that produced the score, not at the log update.

Only create commits when the user asks. If the user hasn't asked yet, prepare the row and the suggested commit message and wait — don't auto-commit.

## `trials.csv` columns

| column | what to put |
|---|---|
| `trial` | 1-based integer; previous row's value + 1 |
| `date` | `YYYY-MM-DD` of the run |
| `commit` | short git SHA (e.g. `git rev-parse --short HEAD`) of the **code** commit that produced the score, not the trial-log commit |
| `algorithm` | model + the hyperparams that differ from baseline. Example: `LightGBM (baseline)`, `LightGBM (lr=0.03, leaves=127)`, `XGBoost (default)` |
| `features` | feature-set delta vs baseline. Example: `baseline`, `baseline + IsLateRace`, `baseline - LapTime_Delta`, `baseline + 3 rolling features` |
| `cv_auc` | OOF AUC printed by the CV cell, 4 decimal places (e.g. `0.9424`). Blank if not computed |
| `lb` | Kaggle public LB score. Blank if not submitted |
| `notes` | anything else worth knowing: surprising fold variance, training time, why the change was tried, what to try next. Quote with `"..."` if the cell contains a comma |

Every column except `lb` and `notes` should be filled. If `cv_auc` is blank, the trial isn't really logged — push back and ask the user to run CV.

## Things to keep aligned

- **The baseline spec lives in `README.md`.** If the user wants to change the baseline itself (e.g. switch CV scheme for everyone going forward), update the README's **Baseline** section in the same commit and call it out — old trial scores under the previous baseline are no longer directly comparable.
- **Match `algorithm` / `features` to what the code actually does.** Don't paraphrase loosely. If the row says `lr=0.03` the committed `params` dict in `training.ipynb` must have `'learning_rate': 0.03`.
- **Don't backfill old trials from memory.** If a row wasn't logged at the time, the commit it points to may no longer match what was scored. Better to leave history thin than to record numbers you can't reproduce.

## Repo conventions

- Data files (`train.csv`, `test.csv`, `sample_submission.csv`, `submission.csv`) are not in git — see `README.md` for how to fetch them via `download.py` / kagglehub.
- The notebook expects all CSVs at the project root (`DATA_DIR = Path('.')`).
- `RANDOM_STATE = 42` everywhere; don't change it without flagging it as a trial.
- `NOTES.md` is a free-form scratchpad for ideas; it's not a substitute for `trials.csv`.

## Running the CV (perf gotchas)

- **Use `num_threads=8` and `force_row_wise=True` in the LightGBM params.** With defaults (or `num_threads=24`), training is ~200× slower on this machine due to thread contention — a 5-fold run that should take ~75s drags out to hours. The score is unchanged in any meaningful way; these are performance-only flags. Don't remove them.
- **Kill any stale Jupyter kernels before running.** `ps aux | grep ipykernel` — old `ipykernel_launcher` processes (from prior nbconvert/IDE sessions) can hog 6+ cores and starve the run. Trial-running from a script (`python3 -u /tmp/run.py`) is more reliable than `jupyter nbconvert --execute` for this exact reason.
- A 5-fold CV run should complete in ~1–2 minutes. If it's been longer than 5 minutes with no fold output, something is wrong (stale kernel, wrong thread count) — kill it and diagnose, don't keep waiting.
