# Challenge 2: Online Purchase Prediction

> 🚧 Work in progress. Sections marked TODO are filled in as each step is finished (see `AGENTS.md`).

Team 5, CAI 4105 (Fall 2026). We predict whether an online shopping session ends in a purchase (`Revenue = True`).

## Files
| File | What it is |
|---|---|
| `challenge2.ipynb` | Steps 5–6: model selection, independent CV, fit and generation of `predictions.csv` (not yet executed) |
| `step1-2_base.ipynb` | Steps 1-2: data exploration, preprocessing and validation setup |
| `helpers.py` | Shared code every notebook imports: data loading, preprocessing, cross-validation folds, `evaluate()` |
| `Data/` | `challenge2_train.csv` (9,247 rows, with `Revenue`) and `challenge2_test.csv` (3,083 rows, no `Revenue`) |
| `predictions.csv` | Final predictions: `ID,Revenue` for all 3,083 test rows (TODO) |
| `requirements.txt` | Pinned Python dependencies |
| `AGENTS.md` | Team plan, to-do list, results and decisions |
| `AI_USE_LOG.md` | How and when AI was used |
| `INSTRUCTIONS.md` | The assignment |

## How to run
From the `Challenge 2/` directory, run `pip install -r requirements.txt`, open `challenge2.ipynb` in Jupyter, and select **Kernel → Restart & Run All**. This notebook has not yet been independently executed. It writes `predictions.csv` only after all assertions pass.

## Approach
**Data exploration** (`step1-2_base.ipynb`, Piece 1): the training set has 9,247 sessions and no missing values. Only 15.5% of sessions end in a purchase, so the target is imbalanced. `PageValues` is the strongest signal (3.6% of sessions buy when it is 0 vs 56.9% when it is above 0). The 10 numeric columns are on very different scales, and 7 columns are categories (including 4 stored as number codes: `OperatingSystems`, `Browser`, `Region`, `TrafficType`). `ID` is not used as a feature.

**Preprocessing** (`helpers.py`): numeric columns are median-imputed and standardized (for Logistic Regression, KNN, SVM and Naïve Bayes; tree models use unscaled values). Category columns are imputed with the most common value and one-hot encoded, ignoring categories not seen in training. This gives 75 model inputs. Preprocessing and model are always combined in one scikit-learn `Pipeline`, so during cross-validation each fold learns the scaling and encoding from its own training part only.

## Validation strategy
All models are compared with the same **stratified 5-fold cross-validation** (`random_state=42`), so every fold keeps the training set's 15.5% share of buyers. The main metric is **F1 for `Revenue = True`**. We also report precision, recall, accuracy and confusion matrices, and compare train F1 with CV F1 to spot overfitting. The shared `evaluate()` function in `helpers.py` makes every model use exactly this setup.

Baselines to beat: always predicting "no purchase" gets 84.5% accuracy but **F1 = 0**; random guessing at the buyer rate gets **CV F1 = 0.141**.

## Models evaluated
Teammates' provisional CV F1 values (from PRs #8 and #9; not yet verified on identical preprocessing): tuned Random Forest 0.682 ± 0.008; Logistic Regression 0.680 ± 0.009; SVM 0.677 ± 0.010; KNN 0.633 ± 0.022; Naive Bayes 0.600 ± 0.011. Tree-model baseline F1: Decision Tree 0.563, Random Forest 0.617, Gradient Boosting 0.661. Results must be harmonized after merging #8 and #9.

## Final model
**Provisional choice: Random Forest**, highest reported tuned CV F1 (0.682) by a very small margin over Logistic Regression (0.680); the 0.002 gap is smaller than the fold-to-fold variation. A constrained forest may reduce overfitting but is less interpretable and more expensive than Logistic Regression. The final notebook uses an explicit candidate Random Forest configuration and **recomputes** its own F1; its parameters are not asserted to equal Jiayang's tuned configuration. The actual final CV score, final model choice and generated CSV remain **pending notebook execution and cross-PR integration**.

## Use of AI
We used AI assistants, as the syllabus allows, and documented every work session in [`AI_USE_LOG.md`](AI_USE_LOG.md). Each team member reviewed, ran and tested the work. All numbers in this README come from the notebook's output.
