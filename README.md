# Week 4: Compare, Cross-Validate and Tune Models

Classification of hospital patient `status` (Admitted / Discharged / Under Treatment) using KNN, Decision Tree, Random Forest and Gradient Boosting, with 5-fold cross-validation, hyper-parameter tuning and an overfitting check.

## Files

| File | Description |
|---|---|
| `Week4_Model_Comparison.ipynb` | Main Google Colab notebook (all 6 steps + Gradient Boosting bonus) |
| `healthcare_data.csv` | Dataset (17,498 patients, 14 columns) |
| `Week4_Report.docx` | Written report with tables, plots and answers |
| `README.md` | This file |

## Dataset

- **Source:** `healthcare_data.csv` (Week 3 dataset)
- **Rows / columns:** 17,498 / 14
- **Target:** `status` (3 classes, balanced: Admitted 5,883, Under Treatment 5,860, Discharged 5,755)
- **Dropped columns:** `patient_id`, `first_name`, `last_name`, `phone`, `email` (identifiers, no predictive meaning)
- **Features used (24 after encoding):** age, bill_amount, length_of_stay, admit_month, admit_year, plus one-hot encoded gender, department, diagnosis, treatment

## How to run (Google Colab)

1. Open [colab.research.google.com](https://colab.research.google.com) and upload `Week4_Model_Comparison.ipynb` (File → Upload notebook).
2. Run the first code cell and upload `healthcare_data.csv` when asked (or put the CSV in the Colab file panel first).
3. Runtime → **Run all**. Total run time is a few minutes (the Gradient Boosting grid is the slowest part).

Required libraries (already in Colab): `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

## What the notebook does

| Step | Content |
|---|---|
| 1 | Clean data, fix invalid dates (discharge before admission), median-fill, one-hot encoding, 80/20 split (`stratify=y`, `random_state=42`) |
| 2 | 5-fold CV of KNN (with `StandardScaler` pipeline), Decision Tree, Random Forest, Gradient Boosting on training data only |
| 3 | KNN: K = 1, 3, ..., 15. Decision Tree: max_depth = 2, 3, 5, 10, None. Plots + explanation |
| 4 | Random Forest `GridSearchCV` (`n_estimators` x `max_depth`, cv=5) = 9 x 5 = 45 trainings |
| Bonus | Gradient Boosting `GridSearchCV` (`learning_rate` x `n_estimators`) = 12 x 5 = 60 trainings |
| 5 | Train / test / gap table for all tuned models |
| 6 | Final model evaluated once on the test set: accuracy, confusion matrix, classification report, decision |

## Results summary

| Step | Result |
|---|---|
| Best CV model (default settings) | Gradient Boosting, 0.3348 |
| Best K (KNN) | 1 (0.3368) |
| Best max_depth (Decision Tree) | 2 (0.3400) |
| Random Forest best params | `max_depth=5`, `n_estimators=100`, CV 0.3427 |
| Gradient Boosting best params | `learning_rate=0.01`, `n_estimators=100`, CV 0.3421 |
| Most overfit model | KNN (K=1), gap 0.66 |
| Most stable model | Decision Tree (depth 2), gap 0.012 |
| Final model | Tuned Random Forest, test accuracy 0.332 |

## Important note

All models score about 0.33, which is chance level for 3 balanced classes. The features contain almost no information about `status` (the data appears randomly generated), so differences between models are within the CV standard deviation. See the report for the full discussion.

## Reproducibility

All random processes use `random_state=42`, so the notebook gives the same numbers on every run.
