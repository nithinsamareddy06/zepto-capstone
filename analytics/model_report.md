# Analytics Module - Modeling Report

Loaded `titanic.csv` (continuation of the single raw load): (891, 15)

## 1. Train/Test Split

Class balance (survived): 
survived
0    0.616
1    0.384

We stratify the split on `survived` because the classes are imbalanced (~62%/38% split). Without stratification, a random split could under- or over-represent the minority (survived) class in train or test, skewing evaluation metrics.

Train shape: (712, 7), Test shape: (179, 7)

## 2. Preprocessing (fit on train only)

Numeric columns (age, fare, sibsp, parch): median-imputed then standard-scaled. Categorical columns (sex, embarked): most-frequent imputed then one-hot encoded. `pclass` is passed through as an ordinal integer. All of this lives inside a single `ColumnTransformer`, fit with `.fit()` only on `X_train`; `X_test` only ever sees `.transform()`, so no information from the test split leaks into fitting.

## 3-4. Classifiers & Evaluation


### Logistic Regression
Confusion matrix:
[[98 12]
 [23 46]]
Accuracy=0.804  Precision=0.793  Recall=0.667  F1=0.724  AUC=0.844

### Decision Tree
Confusion matrix:
[[97 13]
 [26 43]]
Accuracy=0.782  Precision=0.768  Recall=0.623  F1=0.688  AUC=0.801

### Random Forest
Confusion matrix:
[[98 12]
 [21 48]]
Accuracy=0.816  Precision=0.800  Recall=0.696  F1=0.744  AUC=0.832

### Comparison table (classifiers)
| Model               |   Accuracy |   Precision |   Recall |    F1 |   AUC |
|:--------------------|-----------:|------------:|---------:|------:|------:|
| Logistic Regression |      0.804 |       0.793 |    0.667 | 0.724 | 0.844 |
| Decision Tree       |      0.782 |       0.768 |    0.623 | 0.688 | 0.801 |
| Random Forest       |      0.816 |       0.8   |    0.696 | 0.744 | 0.832 |

Saved `charts/decision_tree.png` and `charts/roc_curves.png`.

## 5. Imbalance Handling Comparison (Random Forest)

| Strategy                |   Precision |   Recall |    F1 |
|:------------------------|------------:|---------:|------:|
| Baseline (no handling)  |       0.8   |    0.696 | 0.744 |
| class_weight='balanced' |       0.814 |    0.696 | 0.75  |
| SMOTE (train fold only) |       0.757 |    0.768 | 0.763 |

**Conclusion:** 'SMOTE (train fold only)' achieved the highest F1 (0.763) among the three strategies. class_weight='balanced' and SMOTE both push recall up relative to the baseline by forcing the model to pay more attention to the minority (survived) class, typically at some cost to precision; the baseline tends to favor the majority class and under-predicts survival, which matters for a use case where missing a true survivor (false negative) is the costlier error.

## 6. Hyperparameter Tuning (Random Forest)

Best params: {'classifier__max_depth': 10, 'classifier__max_features': 'sqrt', 'classifier__n_estimators': 100}
Best CV F1 score: 0.749
OOB score of best estimator (refit on full train set): 0.820

## 7. Regression Side-Task — Predicting `fare`

MAE=20.898  RMSE=30.533  R2=0.398  Adjusted R2=0.373
Saved `charts/regression_residuals.png`.
The residual spread visibly widens at higher predicted fares (funnel shape), which indicates **heteroscedasticity** — the model's errors are not uniform across the range of predictions, and it under-predicts extreme high fares in particular (likely first-class outliers).

## 8. Final Model Comparison & Recommendation

### Classification metrics
| Model               |   Accuracy |   Precision |   Recall |    F1 |   AUC |
|:--------------------|-----------:|------------:|---------:|------:|------:|
| Logistic Regression |      0.804 |       0.793 |    0.667 | 0.724 | 0.844 |
| Decision Tree       |      0.782 |       0.768 |    0.623 | 0.688 | 0.801 |
| Random Forest       |      0.816 |       0.8   |    0.696 | 0.744 | 0.832 |

### Regression metrics (separate scale, not comparable to classification)
|                          |    MAE |   RMSE |    R2 |   Adjusted R2 |
|:-------------------------|-------:|-------:|------:|--------------:|
| Linear Regression (fare) | 20.898 | 30.533 | 0.398 |         0.373 |

**Recommendation:** Deploy **Random Forest**. It has the highest F1 score (0.744) among the three classifiers, balancing precision (0.800) and recall (0.696), and its AUC (0.832) is competitive with the others. Random Forest in particular is also more robust to the noisy/imputed features here than a single Decision Tree, without the linear assumptions of Logistic Regression.

Saved the full fitted pipeline (Random Forest) to `best_pipeline.joblib`.
Reload check on raw sample row -> original pred=0, reloaded pred=0, match=True