# Team Context: Challenge 1 (Predict Apartment Rent)

> **Deadline: Sunday, October 11, 2026** (extended from Oct 4).
> Every teammate: give this file to your AI before you start working.
> **`main` is protected: you cannot push to it directly.** Work on a branch and open a pull request (see Git workflow below).

## How to keep this file updated (everyone + every AI follows this)
1. `git checkout main && git pull` before you start, so you have the latest version. Then make your own branch (see Git workflow).
2. When you finish something, check the box in the To-do list below, put your name on it, and add any results or decisions.
   **Also add a line to the AI use log** (required by the syllabus): what AI helped with, and what you did yourself.
3. Commit this file together with your work (`git add AGENTS.md <your files>`), push **your branch**, and open a pull request.
4. **AI assistants:** after completing a task, update the To-do list, Results, Decisions log and AI use log in this file. Don't wait to be asked.
   If you can't edit files (e.g. ChatGPT/Claude in a browser), end your answer with a ready-to-paste AI use log line for the user.
5. Only edit your own lines. If `git pull` reports a conflict here, keep both people's lines.

## Git workflow (main is protected)
Pushing to `main` is blocked. Every change goes through a **branch + pull request (PR)**, and a teammate must approve it before it's merged.

```bash
git checkout main
git pull                                  # 1. get the latest main
git checkout -b yourname-what-you-do      # 2. new branch, e.g. daniel-step3-models
# ...work, then Kernel → Restart & Run All to make sure the notebook still runs...
git add <your files> AGENTS.md            # 3. never `git add .` blindly
git commit -m "Short description of what you did"
git push -u origin yourname-what-you-do   # 4. push YOUR branch, not main
```
5. Open the PR: GitHub shows a **"Compare & pull request"** button, or run `gh pr create --fill`.
6. Tell the team in the group chat; someone reviews and clicks **Merge**.
7. After it's merged: `git checkout main && git pull` before starting anything new.

**Rules**
- Never `git push --force`, and never try to push to `main`. If a push is "rejected", stop and ask the team.
- Don't edit `challenge1.ipynb` at the same time as someone else (notebooks merge badly). Do your work in your own notebook (e.g. `step3_daniel.ipynb`) or tell the team you're editing it.
- Before opening a PR, the notebook must run top to bottom with no errors.
- Reviewers: check that it runs, doesn't use the test file for training/tuning, and has an AI use log line.

**AI assistants:** follow this workflow exactly. Before committing, check the current branch (`git branch --show-current`); if it is `main`, create a new branch first. Never push to `main`, never force-push, never run `git reset --hard` or delete branches without the user's OK. After pushing, give the user the PR link or tell them to open one.

## The task
- Regression: predict `price` (monthly rent) for apartments.
- `Data/challenge1_train.csv`: 7,398 rows, price included. Use it for everything.
- `Data/challenge1_test.csv`: 2,467 rows, price hidden. Use it ONLY for the final predictions.
- Output: `predictions.csv` with exactly 2 columns, `ID,price`, and 2,467 rows.

## Hard rules (from the assignment, we lose points otherwise)
- Use only the training file for development, validation, feature engineering and tuning. Never fit, tune, or peek using the test file; it's only for the final predictions.
- Never use `ID` as a feature.
- Evaluate all 4 required models: LinearRegression, DecisionTreeRegressor, RandomForestRegressor, Gradient Boosting (we use XGBRegressor). Extra regression models are allowed, and the final model can be one of them.
- Don't try to obtain or reconstruct the hidden test prices. No external datasets.
- Don't share code, models or predictions with other teams.
- The final solution must be reproducible.
- Everyone contributes; GitHub activity is the evidence. Make your own commits.

## Team conventions (our choices, to satisfy the rules above)
- `random_state=42` everywhere (reproducibility).
- Preprocessing and the model go in ONE sklearn `Pipeline`, so nothing leaks during CV.
- Same 5-fold CV and RMSE for every model (see Shared setup).

## To-do list (the whole project)
Mark items `[x]` when done and add `(name)`. Claim an item by putting your name next to it before you start.

**Step 1: Look at the data and prepare it**
- [x] Load train.csv; check columns, types, missing values, price distribution (EDA) (Daniel Ibarra)
- [x] Decide which columns are numeric vs categorical (write the lists in Decisions log) (Daniel Ibarra)
- [x] Build the preprocessing `ColumnTransformer` (impute + one-hot, + scaler for linear) (Daniel Ibarra): `make_preprocessor(scale=...)` in notebook

**Step 2: Validation strategy**
- [x] Shared `KFold(5, shuffle=True, random_state=42)` + RMSE scoring (see Shared setup) (Daniel Ibarra)
- [x] Wrap preprocessing + model in a `Pipeline` (Daniel Ibarra): use `evaluate(name, model, scale=...)` in notebook

**Step 3: Four required models**
- [x] Linear Regression: CV RMSE + train RMSE (Joaquin De La Barrera)
- [x] Decision Tree: CV RMSE + train RMSE (Joaquin De La Barrera)
- [x] Random Forest: CV RMSE + train RMSE (Joaquin De La Barrera)
- [x] XGBoost: CV RMSE + train RMSE (Joaquin De La Barrera)
- [x] Comparison table filled in (Results section below) (Joaquin De La Barrera)

**Step 4: Improve**
- [x] GridSearchCV on the best 1-2 models (Jiayang)
- [x] (Optional) feature engineering, e.g. log(price), sqft per bedroom (Jiayang): log(price) via TransformedTargetRegressor

**Step 5: Pick the winner**
- [x] Measure train RMSE for the tuned candidates (to judge overfitting/underfitting) (Kevin Herrera)
- [x] Write a few sentences: validation RMSE, overfitting/underfitting, complexity, computational requirements, interpretability (Kevin Herrera): also re-ran Grid Search on the log(price) XGBoost (Step 4.6)

**Step 6: Final predictions**
- [x] Refit the final model on all of train.csv, predict test.csv (Kevin Herrera)
- [x] Save `predictions.csv` (`ID,price`, one row for every one of the 2,467 test IDs) and check it (Kevin Herrera)

**Submission**
- [x] Final notebook runs top to bottom (Restart & Run All) and clearly shows the whole workflow, from loading data to predictions (Kevin Herrera): ~1 min
- [x] README.md: approach, validation strategy, models evaluated, final model, final validation performance (+ Use of AI section) (Kevin Herrera)
- [x] requirements.txt so the environment can be reproduced (Kevin Herrera): pinned versions of the packages the notebook uses
- [x] Any extra code files the notebook needs are committed (Kevin Herrera): none needed, everything is in the notebook
- [ ] Everyone has their own commits
- [ ] Everyone submits their own `Peer-review form.docx` in Canvas

## AI use policy (from the syllabus)
AI is allowed, but: (1) AI cannot pass the course for us, (2) AI contributions must be attributed and true, (3) AI use must be open and documented.
- **Everyone logs, every work session.** One line per session, using the format below. Log manual work too, so it's clear who did what.
- Be honest: only list what actually happened. Don't claim AI-written code as manual.
- Every claim in the README must match what the notebook actually outputs. No made-up numbers.

### AI use log
Format: `- YYYY-MM-DD (Name, AI or "manual"): what it helped with / what you did`
- 2026-10-10 (Daniel Ibarra, AI): AI wrote notebook Steps 1-2 (EDA, preprocessing, 5-fold CV setup, baseline), drafted AGENTS.md, the simple draw.io plan and .gitignore, and explained the ML concepts.
- 2026-10-10 (Daniel Ibarra, manual): Set up the git repo, ran and checked the notebook in JupyterLab, and organized the team workflow.
- 2026-10-10 (Joaquin De La Barrera, AI): AI added the Step 3 notebook cells (4 baseline models via `evaluate()`, comparison table, takeaways), ran the notebook, updated AGENTS.md, and explained each cell.
- 2026-10-10 (Jiayang, AI): Used AI for Step 4: GridSearchCV tuning for Random Forest and XGBoost, log(price) target with XGBoost, and the Step 4 write-up.
- 2026-10-11 (Daniel Ibarra, AI): AI re-ran Step 4 and checked its numbers.
- 2026-10-11 (Daniel Ibarra, manual): Tested Step 4 in JupyterLab and updated AGENTS.md.
- 2026-10-11 (Daniel Ibarra, AI): AI checked AGENTS.md against the official Canvas instructions and added missing rules and checklist items.
- 2026-10-11 (Kevin Herrera, AI): AI (Claude Code) added Step 4.6 (Grid Search on log(price) XGBoost), Steps 5-6 (candidate table with train/CV RMSE, final model choice, feature importances, predictions.csv + checks), README.md and requirements.txt, and ran the notebook top to bottom.
- 2026-10-11 (Kevin Herrera, manual): Reviewed and approved PR #4 and reviewed the Step 5-6 results.

## Shared setup (everyone uses this, so our numbers are comparable)
```python
from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"   # flip the sign to get RMSE
```
- Numeric columns: SimpleImputer(median) (+ StandardScaler for linear regression)
- Categorical columns: SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown="ignore")

## Who does what
| Person | Task | Status |
|---|---|---|
| Daniel Ibarra | Step 1: EDA + preprocessing pipeline | done |
| Daniel Ibarra | Step 2: CV setup + `evaluate()` helper | done |
| Joaquin De La Barrera | Step 3: 4 baseline models | done |
| Jiayang | Step 4: GridSearch tuning (RF, XGBoost) | done |
| Kevin Herrera | Steps 5-6: pick model, predictions.csv, README, requirements.txt | done |

## Results so far (5-fold CV RMSE, lower is better)
| Model | Train RMSE | CV RMSE (mean ± std) | Notes |
|---|---|---|---|
| Baseline (always guess mean) | 821.4 | 820.7 ± 41.8 | DummyRegressor; every real model must beat this |
| Linear Regression | 520.9 | 526.8 ± 25.3 | scale=True; almost no overfitting but underfits |
| Decision Tree | 40.7 | 562.4 ± 64.8 | default settings; heavy overfitting |
| Random Forest | 163.1 | 426.5 ± 33.6 | default (100 trees) |
| XGBoost | 206.5 | 414.7 ± 22.3 | default |
| Random Forest (tuned) | | 420.96 | GridSearch: 200 trees, max_depth=20, min_samples_split=5 |
| Random Forest (tuned), re-measured in Step 5 | 222.0 | 421.0 ± 31.8 | |
| XGBoost (tuned) | 280.9 | 413.1 ± 23.2 | GridSearch: 200 trees, max_depth=5, learning_rate=0.1 |
| XGBoost (tuned) + log(price) | 307.2 | 404.9 ± 25.0 | Step 4.5 |
| **XGBoost + log(price), re-tuned (FINAL)** | 186.2 | **389.8 ± 26.8** | Step 4.6 GridSearch: 300 trees, max_depth=7, learning_rate=0.1, subsample=0.8 |

## Decisions log (newest at top)
- 2026-10-11 (Kevin Herrera): FINAL MODEL = XGBoost + log(price) re-tuned (CV RMSE 389.8 ± 26.8, train 186.2). Its gap (≈204) is about the same as tuned RF (≈199), but it has the best CV score. predictions.csv made from it (2,467 rows). Full notebook now runs in ~1 min.
- 2026-10-11 (Daniel Ibarra): Step 4 reviewed. Leading candidate for Step 5 is tuned XGBoost + log(price) (CV RMSE 404.86). Train RMSE still needs checking for overfitting. Full notebook takes ~7-8 min to run.
- 2026-10-10 (Joaquin De La Barrera): Step 3 used default hyperparameters for all 4 models. XGBoost (414.7) and Random Forest (426.5) are the best, so they are the ones to tune in Step 4. Decision Tree overfits (train 40.7 vs CV 562.4).
- 2026-10-10 (Daniel Ibarra): Features: NUM_COLS = bathrooms, bedrooms, square_feet, latitude, longitude; CAT_COLS = category, has_photo, state. `ID` excluded.
- 2026-10-10 (Daniel Ibarra): EDA: no missing values in train; imputers kept as a safety net. price is right-skewed ($200 to $8,800, median $1,274), so try log(price) in Step 4. `category` is 99.96% "apartment" (kept, nearly uninformative).
- 2026-10-10 (Daniel Ibarra): Scaling only for linear models (`make_preprocessor(scale=True)`); trees use `scale=False`.

## Course materials
Slides, labs and notes from class are in the `Module1`-`Module5` folders on Canvas (not in this repo). The Lab 4&5 notebook (`Module4&5-Lab4&5_TBM_credit_approval_student.ipynb`) uses the same Pipeline / GridSearchCV pattern as this project.

## Repo layout
- `challenge1.ipynb`: the final notebook; runs top to bottom (Steps 1-6) and writes `predictions.csv`
- `Data/`: the CSVs
- `predictions.csv`, `README.md`, `requirements.txt`
