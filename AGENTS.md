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
- [ ] Linear Regression: CV RMSE + train RMSE
- [ ] Decision Tree: CV RMSE + train RMSE
- [ ] Random Forest: CV RMSE + train RMSE
- [ ] XGBoost: CV RMSE + train RMSE
- [ ] Comparison table filled in (Results section below)

**Step 4: Improve**
- [ ] GridSearchCV on the best 1-2 models
- [ ] (Optional) feature engineering, e.g. log(price), sqft per bedroom

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
Format: `- YYYY-MM-DD (Name, tool or "manual"): what it helped with / what you did`
- 2026-10-10 (Daniel Ibarra, Claude Code): Claude wrote notebook Steps 1-2 (EDA, preprocessing, 5-fold CV setup, baseline), drafted AGENTS.md, the simple draw.io plan and .gitignore, and explained the ML concepts.
- 2026-10-10 (Daniel Ibarra, manual): Set up the git repo, ran and checked the notebook in JupyterLab, and organized the team workflow.

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
| (name) | Step 3: 4 baseline models | not started |
| (name) | Step 4: GridSearch tuning (RF, XGBoost) | not started |
| (name) | Steps 5-6: pick model, predictions.csv, README, requirements.txt | not started |

## Results so far (5-fold CV RMSE, lower is better)
| Model | Train RMSE | CV RMSE (mean ± std) | Notes |
|---|---|---|---|
| Baseline (always guess mean) | 821.4 | 820.7 ± 41.8 | DummyRegressor; every real model must beat this |
| Linear Regression | | | |
| Decision Tree | | | |
| Random Forest | | | |
| XGBoost | | | |

## Decisions log (newest at top)
- 2026-10-10 (Daniel Ibarra): Features: NUM_COLS = bathrooms, bedrooms, square_feet, latitude, longitude; CAT_COLS = category, has_photo, state. `ID` excluded.
- 2026-10-10 (Daniel Ibarra): EDA: no missing values in train; imputers kept as a safety net. price is right-skewed ($200 to $8,800, median $1,274), so try log(price) in Step 4. `category` is 99.96% "apartment" (kept, nearly uninformative).
- 2026-10-10 (Daniel Ibarra): Scaling only for linear models (`make_preprocessor(scale=True)`); trees use `scale=False`.

## Repo layout
- `challenge1.ipynb`: the final notebook; must run top to bottom. Steps 1-2 done; add each model with `evaluate("Name", Model(random_state=42), scale=...)`
- `Data/`: the CSVs
- `predictions.csv`, `README.md`, `requirements.txt`
