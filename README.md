# Challenge 1: Predict Apartment Rent

CAI 4105, Fall 2026, Team 5: Daniel Ibarra, Joaquin De La Barrera, Jiayang, Kevin Herrera

We predict the monthly rent (`price`) of an apartment from its size, location and listing details. Everything is in **`challenge1.ipynb`**, from loading the data to writing **`predictions.csv`**.

## Files
| File | What it is |
|---|---|
| `challenge1.ipynb` | Final notebook: EDA → preprocessing → validation → 4 models → tuning → final model → predictions |
| `predictions.csv` | Final predictions: `ID,price`, 2,467 rows (one per test apartment) |
| `Data/challenge1_train.csv` | Training data (7,398 rows, with price) |
| `Data/challenge1_test.csv` | Test data (2,467 rows, no price), used only in Step 6 |
| `requirements.txt` | Package versions used |
| `AGENTS.md` | Team plan, to-do list, decisions log and AI use log |

## How to reproduce
```bash
pip install -r requirements.txt
jupyter lab challenge1.ipynb
```
Then **Kernel → Restart Kernel and Run All Cells**. It runs in about 1 minute on a laptop and overwrites `predictions.csv` with the same predictions. We used Python 3.14 (Anaconda), and `random_state=42` everywhere.

## Approach
1. **Data preparation:** 7,398 rows, no missing values. Features: 5 numeric (`bathrooms`, `bedrooms`, `square_feet`, `latitude`, `longitude`) and 3 categorical (`category`, `has_photo`, `state`). `ID` is **not** used. Numeric columns get a median imputer (plus `StandardScaler` for Linear Regression only), and categorical columns get a most-frequent imputer + `OneHotEncoder(handle_unknown="ignore")`. Imputers are a safety net in case the test file has blanks. `price` is right-skewed ($200 to $8,800), so we later tried predicting log(price).
2. **Validation strategy:** **5-fold cross-validation** (`KFold(5, shuffle=True, random_state=42)`) on the training file only, with **RMSE** as the metric. Every model uses the same folds. Preprocessing and the model sit in one sklearn `Pipeline`, so the preprocessing is re-fit inside each fold and nothing leaks. We record both **train RMSE** and **CV RMSE**, because a big gap means overfitting. We chose CV over a single train/validation split because with ~7,400 rows it uses all the data for validation and gives a mean ± std instead of one noisy number. Grid Search uses the same folds.
3. **Models:** a mean baseline, then the 4 required models: Linear Regression, Decision Tree, Random Forest and Gradient Boosting (XGBoost).
4. **Improvement:** `GridSearchCV` on Random Forest (trees, max depth, min samples split) and XGBoost (trees, max depth, learning rate). Then a log(price) target with `TransformedTargetRegressor`, and a second Grid Search on the log(price) XGBoost (also tuning `subsample`).
5. **Selection** and **6. Predictions:** see below.

## Results (5-fold CV RMSE in $, lower is better)
| Model | Train RMSE | CV RMSE (mean ± std) |
|---|---|---|
| Baseline (always predict the mean) | 821.4 | 820.7 ± 41.8 |
| Linear Regression | 520.9 | 526.8 ± 25.3 |
| Decision Tree | 40.7 | 562.4 ± 64.8 |
| Random Forest | 163.1 | 426.5 ± 33.7 |
| XGBoost | 206.5 | 414.7 ± 22.3 |
| Random Forest, tuned | 222.0 | 421.0 ± 31.8 |
| XGBoost, tuned | 280.9 | 413.1 ± 23.2 |
| XGBoost, tuned + log(price) | 307.2 | 404.9 ± 25.0 |
| **XGBoost + log(price), re-tuned (final)** | **186.2** | **389.8 ± 26.8** |

What the models show:
- **Linear Regression** underfits: train and CV are almost equal, but both are high. One straight-line formula can't capture how rent depends on location.
- **Decision Tree** overfits badly: it nearly memorizes the training data (train 40.7) but is the worst real model on new data.
- **Random Forest** and **XGBoost** do much better by combining many trees. Tuning helped a little; predicting log(price) and then re-tuning helped the most.

## Final model
**XGBoost on log(price), with preprocessing in the same Pipeline:** 300 trees, `max_depth=7`, `learning_rate=0.1`, `subsample=0.8`, `random_state=42`. Predictions are converted back from log to dollars.

- **Final validation performance: 5-fold CV RMSE = 389.8 ± 26.8** (vs. 820.7 for the mean baseline).
- **Why this model:** it has the lowest CV RMSE of everything we tried, and its fold-to-fold std is similar to the others. It does fit the training data better than new data (train 186 vs. CV 390), but that gap is about the same as the tuned Random Forest's, and it still scores best on the held-out folds. It's fast (seconds to train), so the extra complexity costs little. The trade-off is interpretability: unlike Linear Regression there's no single formula, but feature importances show that location (especially California and other high-cost states) drives the predictions the most.
- **Predictions:** the final model is re-trained on all 7,398 training rows and predicts `challenge1_test.csv`. The notebook checks that `predictions.csv` has exactly the 2,467 test IDs in order, 2 columns (`ID,price`), and no missing or negative prices.

## Use of AI
We used AI assistants (as allowed by the syllabus) to help write notebook code, explain the ML concepts and draft documentation. Each team member reviewed, ran and tested the work. A detailed per-person, per-session log is in the **AI use log** section of [`AGENTS.md`](AGENTS.md). All numbers in this README come from the notebook's output.
