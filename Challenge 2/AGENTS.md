# Team Context: Challenge 2 (Online Purchase Prediction)

> **Deadline: Sunday, November 15, 2026, 11:59 PM.**
> Every teammate: open the `Challenge 2/` folder in your AI tool (or give it this file) before you start working.
> **`main` is protected: you cannot push to it directly.** Work on a branch and open a pull request (see Git workflow).
> Full assignment text: [`INSTRUCTIONS.md`](INSTRUCTIONS.md).

## The 3 files everyone keeps up to date
| File | What it is | When to update it |
|---|---|---|
| `AGENTS.md` (this file) | Plan, to-do list, shared setup, results, decisions | Every time you finish a task |
| [`README.md`](README.md) | How the project works (graded) | Every time your change affects what the README says |
| [`AI_USE_LOG.md`](AI_USE_LOG.md) | When and how AI was used (required by the syllabus) | Every work session where AI helped build the project |

## Rules for every change (everyone + every AI follows this)
1. `git checkout main && git pull` before you start, then make your own branch.
2. When you finish a task: check its box in the To-do list with your name, and add results/decisions below.
3. **Keep the README accurate.** If your change adds, removes or changes anything the README describes
   (data prep, validation, models, results, final model, how to run), update `README.md` in the same pull request.
   Every number in the README must match what the notebook actually prints. No made-up or placeholder numbers in the final version.
4. **Log AI use** in `AI_USE_LOG.md` (see "What to log" there). One line per work session, not per question.
5. Commit those files together with your work (`git add "Challenge 2/<your files>" "Challenge 2/AGENTS.md" ...`), push **your branch**, open a pull request.
6. Only edit your own lines in the log and to-do list. If `git pull` shows a conflict in these files, keep both people's lines.

**AI assistants:** after completing a task, update the To-do list, Results, Decisions log, `README.md` (if affected) and `AI_USE_LOG.md` without waiting to be asked.
If you can't edit files (e.g. ChatGPT or Claude in a browser), end your answer with a ready-to-paste `AI_USE_LOG.md` line and a note on which README section needs changing.

## Git workflow (main is protected)
```bash
git checkout main
git pull                                  # 1. get the latest main
git checkout -b yourname-what-you-do      # 2. e.g. joaquin-linear-models
# ...work, then Kernel → Restart & Run All to make sure your notebook runs...
git add "Challenge 2/<your files>" "Challenge 2/AGENTS.md" "Challenge 2/AI_USE_LOG.md" "Challenge 2/README.md"
git commit -m "Short description of what you did"
git push -u origin yourname-what-you-do   # 3. push YOUR branch, not main
gh pr create --fill                       # 4. or click "Compare & pull request" on GitHub
```
Tell the team in the group chat; someone reviews and merges. After the merge: `git checkout main && git pull`.

- Never `git push --force`, never push to `main`. If a push is rejected, stop and ask the team.
- Work in **your own notebook** (names in "Who does what"); the finisher merges them into `challenge2.ipynb`.
- Reviewers check: notebook runs, test file not used for training/tuning, README still accurate, AI log line present.

**AI assistants:** check `git branch --show-current` before committing; if it's `main`, make a branch first. Never push to `main`, force-push, `git reset --hard`, or delete branches without the user's OK.

## The task
- Classification: predict `Revenue` (True = the shopping session ended in a purchase).
- `Data/challenge2_train.csv`: 9,247 rows, `Revenue` included (7,816 False / 1,431 True, about 15% buyers). Use it for everything.
- `Data/challenge2_test.csv`: 3,083 rows, `Revenue` hidden. Use it ONLY for the final predictions.
- Output: `predictions.csv` with exactly 2 columns, `ID,Revenue` (True/False), 3,083 rows.
- **Primary metric: F1 score for the positive class (`Revenue=True`).** The data is imbalanced, so never judge a model by accuracy alone.

## Hard rules (from the assignment, we lose points otherwise)
- Use only the training file for development, validation, feature engineering and tuning. The test file is only for the final predictions.
- Never use `ID` as a feature.
- Evaluate all 7 required models: Logistic Regression, Naïve Bayes, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting. Extra models are allowed, and the final model can be one of them.
- Use Grid Search to tune selected models.
- Don't try to obtain or reconstruct the hidden test labels. No external datasets (including the original public version of this dataset).
- Don't share code, models or predictions with other teams.
- The final solution must be reproducible.
- Everyone contributes; GitHub activity is the evidence. Make your own commits.

## Team conventions
- `random_state=42` everywhere.
- Preprocessing + model go in ONE sklearn `Pipeline`, so nothing leaks during cross-validation.
- Everyone uses the shared helper from Step 1-2 (same preprocessing, same folds, same scoring), so results are comparable.

## Shared setup (Step 1-2 will turn this into a helper everyone imports)
```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # keeps ~15% buyers in every fold
scoring = "f1"            # with y = Revenue as 0/1, this is F1 for Revenue=True
```
Proposed column groups (to be confirmed by the Step 1 EDA):
- Numeric: Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay
- Categorical (one-hot): Month, VisitorType, Weekend, OperatingSystems, Browser, Region, TrafficType (these last four are numbered categories, not amounts)
- Excluded: ID. Target: Revenue

## To-do list
Mark items `[x]` when done and add `(name)`. Claim an item by putting your name next to it before you start.

**Step 1: Explore and prepare the data** (Daniel Ibarra)
- [ ] EDA: columns, types, missing values, target balance, key patterns
- [ ] Confirm numeric vs categorical lists (write them in Decisions log)
- [ ] Preprocessing `ColumnTransformer` (scale numeric for linear/distance models, one-hot categoricals)

**Step 2: Validation strategy** (Daniel Ibarra)
- [ ] Shared `StratifiedKFold` + F1 scoring + `evaluate()` helper reporting F1, precision, recall, train vs CV F1
- [ ] Baseline (DummyClassifier) so every model has something to beat

**Step 3: Seven required models**
- [ ] Logistic Regression, Naïve Bayes, KNN, SVM (Person 2: ____)
- [ ] Decision Tree, Random Forest, Gradient Boosting (Person 3: ____)
- [ ] Comparison table filled in (Results section)

**Step 4: Improve**
- [ ] GridSearchCV on selected models (Persons 2 and 3, 2 models each)
- [ ] (Optional) feature engineering / class_weight / decision threshold (Person 4: ____)

**Step 5: Pick the final model** (Person 4: ____)
- [ ] Short justification: validation F1, overfitting/underfitting, complexity, compute, interpretability

**Step 6: Predictions + submission** (Person 4: ____)
- [ ] Refit final model on all of train, predict test, save `predictions.csv` (`ID,Revenue`, 3,083 rows) and check it
- [ ] Final `challenge2.ipynb` runs top to bottom (Restart & Run All), from loading data to predictions
- [ ] `README.md` complete and accurate; `requirements.txt` with pinned versions
- [ ] Everyone has their own commits
- [ ] Everyone submits their own `Peer-review form.docx` in Canvas

## Who does what
| Person | Task | Notebook | Status |
|---|---|---|---|
| Daniel Ibarra | Steps 1-2: EDA, preprocessing, validation, shared helper | `step1-2_base.ipynb` | in progress |
| ____ | Steps 3-4: LogReg, Naïve Bayes, KNN, SVM + Grid Search | `step3_linear.ipynb` | |
| ____ | Steps 3-4: Decision Tree, Random Forest, Gradient Boosting + Grid Search | `step3_trees.ipynb` | |
| ____ | Steps 4-6: feature engineering, final model, predictions, README, requirements | `challenge2.ipynb` | |

## Results (5-fold stratified CV, F1 for Revenue=True, higher is better)
| Model | Train F1 | CV F1 (mean ± std) | Precision | Recall | Notes |
|---|---|---|---|---|---|

## Decisions log (newest at top)
Format: `- YYYY-MM-DD (Name): decision and why`
