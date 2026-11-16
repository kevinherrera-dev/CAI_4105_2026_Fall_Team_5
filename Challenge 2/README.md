# Challenge 2: Online Purchase Prediction

Team 5, CAI 4105 (Fall 2026): Daniel Ibarra, Joaquin De La Barrera, Jiayang Li, Kevin Herrera.

We predict whether an online shopping session ends in a purchase (`Revenue = True`). The final workflow, from loading the data to writing `predictions.csv`, is in **`challenge2.ipynb`**.

## Files
| File | What it is |
|---|---|
| `challenge2.ipynb` | **Final notebook:** EDA summary, validation setup, 7-model comparison, final model, CV check, predictions |
| `predictions.csv` | Final predictions: `ID,Revenue` (True/False) for all 3,083 test rows, in the test file's order |
| `helpers.py` | Shared code every notebook imports: data loading, preprocessing, CV folds, `evaluate()` |
| `step1-2_base.ipynb` | Steps 1-2 in full: data exploration, preprocessing and validation setup |
| `step3_linear.ipynb` | Steps 3-4: Logistic Regression, Naive Bayes, KNN, SVM + Grid Search |
| `step3_trees.ipynb` | Steps 3-4: Decision Tree, Random Forest, Gradient Boosting + Grid Search |
| `results/` | Saved result tables from the step notebooks |
| `Data/` | `challenge2_train.csv` (9,247 rows, with `Revenue`) and `challenge2_test.csv` (3,083 rows, no `Revenue`) |
| `requirements.txt` | Pinned Python dependencies |
| `AGENTS.md`, `AI_USE_LOG.md` | Team plan and decisions; how AI was used |

## How to reproduce
```bash
cd "Challenge 2"
pip install -r requirements.txt
jupyter notebook challenge2.ipynb
```
Then **Kernel → Restart & Run All** (about 15 seconds). It rewrites `predictions.csv` with identical predictions. We used Python 3.11 and `random_state=42` everywhere.

## Approach
**Data exploration** (`step1-2_base.ipynb`): 9,247 sessions and no missing values. Only 15.5% of sessions end in a purchase, so the target is imbalanced. `PageValues` is the strongest signal (3.6% of sessions buy when it is 0 vs 56.9% when it is above 0). The 10 numeric columns are on very different scales, and 7 columns are categories (including 4 stored as number codes: `OperatingSystems`, `Browser`, `Region`, `TrafficType`). `ID` is not used as a feature.

**Preprocessing** (`helpers.py`): numeric columns are median-imputed and standardized for Logistic Regression, KNN, SVM and Naive Bayes; tree models use unscaled values. Categorical columns are imputed with the most common value and one-hot encoded, ignoring categories not seen in training (75 model inputs). **Feature engineering:** for the linear and distance models we also applied `log1p` to the skewed counts, durations and `PageValues`, which raised CV F1 by about 0.12 for Logistic Regression and KNN. Preprocessing and model are always combined in one scikit-learn `Pipeline`, so in cross-validation each fold learns the scaling and encoding from its own training part only.

## Validation strategy
All models are compared with the same **stratified 5-fold cross-validation** (`random_state=42`) on the training file only, so every fold keeps the 15.5% share of buyers. The main metric is **F1 for `Revenue = True`**. We also report precision, recall, accuracy and confusion matrices, and compare train F1 with CV F1 to spot overfitting. Grid Search uses the same folds and scoring. Baselines: always predicting "no purchase" gets 84.5% accuracy but **F1 = 0**; a stratified random guess gets **CV F1 = 0.141**.

## Models evaluated (5-fold stratified CV, F1 for Revenue = True)
| Model | Train F1 | CV F1 (mean ± std) | Best settings (Grid Search) |
|---|---|---|---|
| **Random Forest (tuned)** | 0.785 | **0.682 ± 0.008** | class_weight=balanced, max_depth=None, min_samples_leaf=5 |
| Logistic Regression (tuned) | 0.679 | 0.680 ± 0.009 | C=0.03, L1, class_weight=balanced, log1p features |
| SVM RBF (tuned) | 0.678 | 0.677 ± 0.010 | C=1, gamma=0.003, class_weight=balanced, log1p features |
| Gradient Boosting (tuned) | 0.720 | 0.670 ± 0.020 | n_estimators=150, learning_rate=0.05, max_depth=3 |
| KNN (tuned) | 0.647 | 0.633 ± 0.022 | k=31, uniform, p=2, log1p features |
| Naive Bayes (tuned) | 0.602 | 0.600 ± 0.011 | var_smoothing=1.0, log1p features |
| Decision Tree (default) | 1.000 | 0.563 ± 0.024 | default (overfits) |
| Baseline (stratified guess) | 0.157 | 0.141 ± 0.015 | DummyClassifier |

Before tuning: Random Forest 0.617 (train F1 1.000), Gradient Boosting 0.661, SVM 0.650, Logistic Regression 0.633, KNN 0.607, Naive Bayes 0.432.

## Final model
**Tuned Random Forest** (100 trees, `class_weight='balanced'`, `min_samples_leaf=5`, `random_state=42`).
- **Validation performance:** the highest CV F1 (0.682). Logistic Regression (0.680) and SVM (0.677) are within one standard deviation, so they are close alternatives.
- **Overfitting:** the default forest memorized the training data (train F1 1.000 vs CV 0.617). `min_samples_leaf=5` cut the gap to about 0.10, and its fold-to-fold std (0.008) is the lowest of all models.
- **Cost and interpretability:** it is less interpretable than Logistic Regression but still cheap (about 2 s to fit). SVM was the slowest to tune (324 s).

**Final validation performance** (re-measured in `challenge2.ipynb`): **CV F1 = 0.6818 ± 0.0081**, precision 0.603, recall 0.785, accuracy 0.887, train F1 0.785. Out-of-fold confusion matrix: 1,123 buyers found, 308 missed, 740 false alarms, 7,076 correct non-buyers. The model was then refit on all 9,247 training rows and predicts 615 purchases among the 3,083 test sessions.

## Use of AI
We used AI assistants, as the syllabus allows, and documented every work session in [`AI_USE_LOG.md`](AI_USE_LOG.md). Each team member reviewed, ran and tested the work. All numbers in this README come from the notebooks' outputs.
