# Challenge 2: Online Purchase Prediction

> 🚧 Work in progress. Sections marked TODO are filled in as each step is finished (see `AGENTS.md`).

Team 5, CAI 4105 (Fall 2026). We predict whether an online shopping session ends in a purchase (`Revenue = True`).

## Files
| File | What it is |
|---|---|
| `challenge2.ipynb` | Final notebook: full workflow from loading the data to `predictions.csv` (TODO) |
| `Data/` | `challenge2_train.csv` (9,247 rows, with `Revenue`) and `challenge2_test.csv` (3,083 rows, no `Revenue`) |
| `predictions.csv` | Final predictions: `ID,Revenue` for all 3,083 test rows (TODO) |
| `requirements.txt` | Package versions to reproduce the environment (TODO) |
| `AGENTS.md` | Team plan, to-do list, results and decisions |
| `AI_USE_LOG.md` | How and when AI was used |
| `INSTRUCTIONS.md` | The assignment |

## How to run
TODO (environment setup + "Restart & Run All" on `challenge2.ipynb`)

## Approach
TODO: data preparation and preprocessing

## Validation strategy
TODO: 5-fold stratified cross-validation, F1 for `Revenue=True` as the main metric

## Models evaluated
TODO: comparison table of the 7 required models (+ any extras)

## Final model
TODO: which model, why, and its final validation performance

## Use of AI
We used AI assistants, as the syllabus allows, and documented every work session in [`AI_USE_LOG.md`](AI_USE_LOG.md). Each team member reviewed, ran and tested the work. All numbers in this README come from the notebook's output.
