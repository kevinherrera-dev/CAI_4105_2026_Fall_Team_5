# Team Context: Challenge 1 (Predict Apartment Rent)

> **Deadline: Sunday, October 11, 2026** (extended from Oct 4).
> Every teammate: give this file to your AI before you start working.

## How to keep this file updated (everyone + every AI follows this)
1. `git pull` before you start, so you have the latest version.
2. When you finish something, check the box in the To-do list below, put your name on it, and add any results or decisions.
   **Also add a line to the AI use log** (required by the syllabus): what AI helped with, and what you did yourself.
3. Commit this file together with your work (`git add AGENTS.md <your files>`), then push.
4. **AI assistants:** after completing a task, update the To-do list, Results, Decisions log and AI use log in this file. Don't wait to be asked.
   If you can't edit files (e.g. ChatGPT/Claude in a browser), end your answer with a ready-to-paste AI use log line for the user.
5. Only edit your own lines. If `git pull` reports a conflict here, keep both people's lines.

## The task
- Regression: predict `price` (monthly rent) for apartments.
- `Data/challenge1_train.csv`: 7,398 rows, price included. Use it for everything.
- `Data/challenge1_test.csv`: 2,467 rows, price hidden. Use it ONLY for the final predictions.
- Output: `predictions.csv` with exactly 2 columns, `ID,price`, and 2,467 rows.

## Hard rules (we lose points otherwise)
- Never fit, tune, or peek using the test file.
- Never use `ID` as a feature.
- Evaluate all 4 models: LinearRegression, DecisionTreeRegressor, RandomForestRegressor, XGBRegressor.
- `random_state=42` everywhere.
- Preprocessing and the model go in ONE sklearn `Pipeline`, so nothing leaks during CV.
- Make your own commits; commits count as proof of participation.

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
- [x] GridSearchCV on the best 1-2 models (Kevin Herrera)
- [x] (Optional) feature engineering, e.g. log(price), sqft per bedroom (Kevin Herrera): log(price) via TransformedTargetRegressor

**Step 5: Pick the winner**
- [ ] Write a few sentences: RMSE, overfitting gap, complexity, speed, interpretability

**Step 6: Final predictions**
- [ ] Refit winner on all of train.csv, predict test.csv
- [ ] Save `predictions.csv` (`ID,price`, 2,467 rows) and check its shape

**Submission**
- [ ] Final notebook runs top to bottom (Restart & Run All)
- [ ] README.md: approach, validation, models, final model, final CV RMSE, Use of AI section
- [ ] requirements.txt (`pip freeze > requirements.txt`)
- [ ] Everyone has their own commits
- [ ] Everyone submits their own peer-review form

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
- 2026-10-11 (Kevin Herrera): Did Step 4: GridSearchCV tuning for Random Forest and XGBoost, plus log(price) on XGBoost.
- 2026-10-11 (Daniel Ibarra, AI): AI re-ran Step 4 and checked its numbers.
- 2026-10-11 (Daniel Ibarra, manual): Tested Step 4 in JupyterLab and updated AGENTS.md.

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
| Kevin Herrera | Step 4: GridSearch tuning (RF, XGBoost) | done |
| (name) | Steps 5-6: pick model, predictions.csv, README, requirements.txt | not started |

## Results so far (5-fold CV RMSE, lower is better)
| Model | Train RMSE | CV RMSE (mean ± std) | Notes |
|---|---|---|---|
| Baseline (always guess mean) | 821.4 | 820.7 ± 41.8 | DummyRegressor; every real model must beat this |
| Linear Regression | 520.9 | 526.8 ± 25.3 | scale=True; almost no overfitting but underfits |
| Decision Tree | 40.7 | 562.4 ± 64.8 | default settings; heavy overfitting |
| Random Forest | 163.1 | 426.5 ± 33.6 | default (100 trees) |
| XGBoost | 206.5 | 414.7 ± 22.3 | default |
| Random Forest (tuned) | | 420.96 | GridSearch: 200 trees, max_depth=20, min_samples_split=5 |
| XGBoost (tuned) | | 413.07 | GridSearch: 200 trees, max_depth=5, learning_rate=0.1 |
| XGBoost (tuned) + log(price) | | 404.86 | best so far; train RMSE not measured yet |

## Decisions log (newest at top)
- 2026-10-11 (Daniel Ibarra): Step 4 reviewed. Leading candidate for Step 5 is tuned XGBoost + log(price) (CV RMSE 404.86). Train RMSE still needs checking for overfitting. Full notebook takes ~7-8 min to run.
- 2026-10-10 (Joaquin De La Barrera): Step 3 used default hyperparameters for all 4 models. XGBoost (414.7) and Random Forest (426.5) are the best, so they are the ones to tune in Step 4. Decision Tree overfits (train 40.7 vs CV 562.4).
- 2026-10-10 (Daniel Ibarra): Features: NUM_COLS = bathrooms, bedrooms, square_feet, latitude, longitude; CAT_COLS = category, has_photo, state. `ID` excluded.
- 2026-10-10 (Daniel Ibarra): EDA: no missing values in train; imputers kept as a safety net. price is right-skewed ($200 to $8,800, median $1,274), so try log(price) in Step 4. `category` is 99.96% "apartment" (kept, nearly uninformative).
- 2026-10-10 (Daniel Ibarra): Scaling only for linear models (`make_preprocessor(scale=True)`); trees use `scale=False`.

## Repo layout
- `challenge1.ipynb`: the final notebook; must run top to bottom. Steps 1-2 done; add each model with `evaluate("Name", Model(random_state=42), scale=...)`
- `Data/`: the CSVs
- `predictions.csv`, `README.md`, `requirements.txt`
