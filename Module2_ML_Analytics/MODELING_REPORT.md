# Modeling Report — Titanic Survival Prediction

## 1. Train/test split

Class balance of `survived`: {0: 0.618, 1: 0.382} — roughly 62/38, which is imbalanced enough that a plain random split risks under- or over-representing the minority (survived=1) class in either split by chance. **Stratifying on `survived`** guarantees both the train and test sets keep this same ~62/38 ratio, so test metrics are not skewed by an accidental class-balance mismatch.

Train size: 711, Test size: 178

## 2. Preprocessing (fit on training data only)

- Numeric features (`age`, `sibsp`, `parch`, `fare`): median-imputed, then `StandardScaler`-scaled.
- Categorical features (`pclass`, `sex`, `embarked`): most-frequent-imputed, then one-hot encoded.
- All of this is wrapped in a `ColumnTransformer` inside a `Pipeline`, so `.fit()` only ever sees `X_train`, and `X_test` only ever goes through `.transform()` — the split happens *before* any preprocessing object is fit, enforcing no test-set leakage structurally.

## 3. Classifiers

### Logistic Regression

Confusion matrix:
```
[[98 12]
 [21 47]]
```

### Decision Tree

Confusion matrix:
```
[[97 13]
 [32 36]]
```

### Random Forest

Confusion matrix:
```
[[98 12]
 [22 46]]
```

### Classifier comparison table
| Model               |   Accuracy |   Precision |   Recall |    F1 |   AUC |
|:--------------------|-----------:|------------:|---------:|------:|------:|
| Logistic Regression |      0.815 |       0.797 |    0.691 | 0.74  | 0.86  |
| Decision Tree       |      0.747 |       0.735 |    0.529 | 0.615 | 0.826 |
| Random Forest       |      0.809 |       0.793 |    0.676 | 0.73  | 0.824 |

## 4. Imbalance-handling comparison (Random Forest)

Class balance: {0: 0.618, 1: 0.382} (survived=1 is the minority class)

| Strategy                |   Precision |   Recall |    F1 |
|:------------------------|------------:|---------:|------:|
| Baseline (no handling)  |       0.793 |    0.676 | 0.73  |
| class_weight='balanced' |       0.754 |    0.721 | 0.737 |
| SMOTE (train fold only) |       0.758 |    0.735 | 0.746 |

**Conclusion:** `SMOTE (train fold only)` achieved the best F1 (0.746) on this split. Titanic's imbalance (~62/38) is mild, so gains over baseline are modest either way; `class_weight='balanced'` tends to nudge recall for the minority class up at a small precision cost without changing the training data, while SMOTE synthesizes new minority examples and can push recall further but risks slightly more overfit synthetic patterns on a dataset this small.

## 5. Hyperparameter tuning (GridSearchCV + OOB score)

Best params: `{'clf__max_depth': None, 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}`

Best CV F1: 0.744

OOB score of best estimator (refit on full training fold): 0.797

## 6. Regression side-task — predicting `fare`

MAE=17.88, RMSE=40.49, R²=0.385, Adjusted R²=0.360

**Heteroscedasticity check:** the residual spread visibly widens at higher predicted fares (a fan/cone shape rather than a uniform band), so this residual plot **does show heteroscedasticity** — the linear model's errors are not constant-variance across the fare range, likely because a few very high first-class fares are disproportionately hard to predict from the available features.

## 7. Final model comparison & recommendation

### Classification metrics (one scale)
| Model               |   Accuracy |   Precision |   Recall |    F1 |   AUC |
|:--------------------|-----------:|------------:|---------:|------:|------:|
| Logistic Regression |      0.815 |       0.797 |    0.691 | 0.74  | 0.86  |
| Decision Tree       |      0.747 |       0.735 |    0.529 | 0.615 | 0.826 |
| Random Forest       |      0.809 |       0.793 |    0.676 | 0.73  | 0.824 |

### Regression metrics (a different scale — not directly comparable to the above)
|   MAE |   RMSE |    R2 |   Adjusted_R2 |
|------:|-------:|------:|--------------:|
| 17.88 |  40.49 | 0.385 |          0.36 |

**Recommendation:** deploy **Logistic Regression**. It posts the best F1 (0.74) among the three classifiers, balancing precision (0.797) and recall (0.691) rather than optimizing only one, and its AUC of 0.86 is the strongest (or near-strongest) of the three, showing good class separation across thresholds. Random Forest (F1=0.73, AUC=0.824) is a close second and remains a reasonable fallback if feature-importance-based interpretability is a priority, while the single Decision Tree (F1=0.615) trails both, consistent with a lone tree overfitting/underfitting more than an ensemble or a regularized linear model on a dataset this size.

## 8. Saved artifact

Saved the complete fitted pipeline for **Logistic Regression** (preprocessing + estimator together) to `best_pipeline.joblib`.

Reload check on one raw test row: original pipeline predicted `0`, reloaded pipeline predicted `0` — **match: True**.
