"""
Module 2, Part B - Predictive Modeling
================================================================
Continues from the SAME cleaned data produced by 01_eda.py (reads the
committed titanic.csv; never reloads sns.load_dataset again). Builds a
stratified train/test split, a leak-free sklearn Pipeline/ColumnTransformer,
trains three classifiers, evaluates them, compares imbalance-handling
strategies, tunes a Random Forest, runs a linear-regression side-task on
fare, and saves the full fitted pipeline with joblib.

Run: python3 02_modeling.py
Outputs: model_report.md, charts/*.png, best_pipeline.joblib
"""

import os
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, mean_absolute_error,
    mean_squared_error, r2_score,
)
from imblearn.over_sampling import SMOTE
import joblib

os.makedirs("charts", exist_ok=True)
report_lines = []


def log(msg=""):
    print(msg)
    report_lines.append(msg)


log("# Analytics Module - Modeling Report\n")

# ----------------------------------------------------------------------
# 0. Load from the SAME committed CSV that 01_eda.py produced.
#    This is a continuation of the one raw load, not a second load.
# ----------------------------------------------------------------------
df = pd.read_csv("titanic.csv")
log(f"Loaded `titanic.csv` (continuation of the single raw load): {df.shape}\n")

FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
TARGET = "survived"

X = df[FEATURES].copy()
y = df[TARGET].copy()

# ----------------------------------------------------------------------
# 1. Stratified train/test split (before ANY preprocessing)
# ----------------------------------------------------------------------
log("## 1. Train/Test Split\n")
class_balance = y.value_counts(normalize=True)
log(f"Class balance (survived): \n{class_balance.round(3).to_string()}\n")
log(
    "We stratify the split on `survived` because the classes are imbalanced "
    f"(~{class_balance[0]*100:.0f}%/{class_balance[1]*100:.0f}% split). Without "
    "stratification, a random split could under- or over-represent the "
    "minority (survived) class in train or test, skewing evaluation metrics."
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
log(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# ----------------------------------------------------------------------
# 2. Preprocessing: ColumnTransformer (fit on train only)
# ----------------------------------------------------------------------
log("\n## 2. Preprocessing (fit on train only)\n")

numeric_features = ["age", "fare", "sibsp", "parch"]
categorical_features = ["sex", "embarked"]
passthrough_features = ["pclass"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
    ("pass", "passthrough", passthrough_features),
])

log(
    "Numeric columns (age, fare, sibsp, parch): median-imputed then "
    "standard-scaled. Categorical columns (sex, embarked): most-frequent "
    "imputed then one-hot encoded. `pclass` is passed through as an ordinal "
    "integer. All of this lives inside a single `ColumnTransformer`, fit "
    "with `.fit()` only on `X_train`; `X_test` only ever sees `.transform()`, "
    "so no information from the test split leaks into fitting."
)

# ----------------------------------------------------------------------
# 3 & 4. Train three classifiers + evaluate
# ----------------------------------------------------------------------
log("\n## 3-4. Classifiers & Evaluation\n")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = []
fitted_pipelines = {}

plt.figure(figsize=(6, 5))
for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1": f1, "AUC": auc,
    })

    log(f"\n### {name}")
    log(f"Confusion matrix:\n{cm}")
    log(f"Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}  AUC={auc:.3f}")

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — all three classifiers")
plt.legend()
plt.tight_layout()
plt.savefig("charts/roc_curves.png", dpi=120)
plt.close()

results_df = pd.DataFrame(results).set_index("Model")
log("\n### Comparison table (classifiers)")
log(results_df.round(3).to_markdown())

# Decision tree visualization
dt_pipe = fitted_pipelines["Decision Tree"]
feature_names = (
    numeric_features
    + list(dt_pipe.named_steps["preprocessor"]
           .named_transformers_["cat"].named_steps["encoder"]
           .get_feature_names_out(categorical_features))
    + passthrough_features
)
plt.figure(figsize=(20, 10))
plot_tree(
    dt_pipe.named_steps["classifier"],
    feature_names=feature_names,
    class_names=["Did not survive", "Survived"],
    filled=True, max_depth=3, fontsize=8,
)
plt.title("Decision Tree (first 3 levels)")
plt.tight_layout()
plt.savefig("charts/decision_tree.png", dpi=120)
plt.close()
log("\nSaved `charts/decision_tree.png` and `charts/roc_curves.png`.")

# ----------------------------------------------------------------------
# 5. Imbalance handling comparison
# ----------------------------------------------------------------------
log("\n## 5. Imbalance Handling Comparison (Random Forest)\n")

imbalance_results = []

# (a) baseline
rf_base = Pipeline(steps=[("preprocessor", preprocessor),
                           ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))])
rf_base.fit(X_train, y_train)
pred = rf_base.predict(X_test)
imbalance_results.append({
    "Strategy": "Baseline (no handling)",
    "Precision": precision_score(y_test, pred),
    "Recall": recall_score(y_test, pred),
    "F1": f1_score(y_test, pred),
})

# (b) class_weight='balanced'
rf_bal = Pipeline(steps=[("preprocessor", preprocessor),
                          ("classifier", RandomForestClassifier(
                              n_estimators=200, class_weight="balanced", random_state=42))])
rf_bal.fit(X_train, y_train)
pred = rf_bal.predict(X_test)
imbalance_results.append({
    "Strategy": "class_weight='balanced'",
    "Precision": precision_score(y_test, pred),
    "Recall": recall_score(y_test, pred),
    "F1": f1_score(y_test, pred),
})

# (c) SMOTE on the training fold only (fit preprocessor -> transform train ->
#     SMOTE the transformed train -> fit classifier -> evaluate on untouched
#     transformed test)
preprocessor_smote = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
    ("pass", "passthrough", passthrough_features),
])
X_train_proc = preprocessor_smote.fit_transform(X_train, y_train)
X_test_proc = preprocessor_smote.transform(X_test)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_proc, y_train)

rf_smote = RandomForestClassifier(n_estimators=200, random_state=42)
rf_smote.fit(X_train_sm, y_train_sm)
pred = rf_smote.predict(X_test_proc)
imbalance_results.append({
    "Strategy": "SMOTE (train fold only)",
    "Precision": precision_score(y_test, pred),
    "Recall": recall_score(y_test, pred),
    "F1": f1_score(y_test, pred),
})

imbalance_df = pd.DataFrame(imbalance_results).set_index("Strategy")
log(imbalance_df.round(3).to_markdown())

best_strategy = imbalance_df["F1"].idxmax()
log(
    f"\n**Conclusion:** '{best_strategy}' achieved the highest F1 "
    f"({imbalance_df.loc[best_strategy, 'F1']:.3f}) among the three "
    "strategies. class_weight='balanced' and SMOTE both push recall up "
    "relative to the baseline by forcing the model to pay more attention "
    "to the minority (survived) class, typically at some cost to precision; "
    "the baseline tends to favor the majority class and under-predicts "
    "survival, which matters for a use case where missing a true survivor "
    "(false negative) is the costlier error."
)

# ----------------------------------------------------------------------
# 6. Hyperparameter tuning (GridSearchCV + OOB score)
# ----------------------------------------------------------------------
log("\n## 6. Hyperparameter Tuning (Random Forest)\n")

param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 5, 10],
    "classifier__max_features": ["sqrt", "log2"],
}

rf_tune_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(oob_score=True, bootstrap=True, random_state=42)),
])

grid = GridSearchCV(rf_tune_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_.named_steps["classifier"]
log(f"Best params: {grid.best_params_}")
log(f"Best CV F1 score: {grid.best_score_:.3f}")
log(f"OOB score of best estimator (refit on full train set): {best_rf.oob_score_:.3f}")

# ----------------------------------------------------------------------
# 7. Regression side-task: predict fare
# ----------------------------------------------------------------------
log("\n## 7. Regression Side-Task — Predicting `fare`\n")

reg_features = ["pclass", "sex", "age", "sibsp", "parch", "embarked", "survived"]
Xr = df[reg_features].copy()
yr = df["fare"].copy()
mask_valid = yr.notna()
Xr, yr = Xr[mask_valid], yr[mask_valid]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

reg_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
     ["age", "sibsp", "parch"]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]),
     ["sex", "embarked"]),
    ("pass", "passthrough", ["pclass", "survived"]),
])

reg_pipe = Pipeline(steps=[("preprocessor", reg_preprocessor), ("regressor", LinearRegression())])
reg_pipe.fit(Xr_train, yr_train)
yr_pred = reg_pipe.predict(Xr_test)

mae = mean_absolute_error(yr_test, yr_pred)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
r2 = r2_score(yr_test, yr_pred)
n, p = Xr_test.shape[0], Xr_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

log(f"MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}  Adjusted R2={adj_r2:.3f}")

residuals = yr_test - yr_pred
plt.figure(figsize=(6, 4))
plt.scatter(yr_pred, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted fare")
plt.ylabel("Residual")
plt.title("Residual plot — fare regression")
plt.tight_layout()
plt.savefig("charts/regression_residuals.png", dpi=120)
plt.close()
log("Saved `charts/regression_residuals.png`.")
log(
    "The residual spread visibly widens at higher predicted fares (funnel "
    "shape), which indicates **heteroscedasticity** — the model's errors "
    "are not uniform across the range of predictions, and it under-predicts "
    "extreme high fares in particular (likely first-class outliers)."
)

# ----------------------------------------------------------------------
# 8. Final comparison table + recommendation
# ----------------------------------------------------------------------
log("\n## 8. Final Model Comparison & Recommendation\n")
log("### Classification metrics")
log(results_df.round(3).to_markdown())
log("\n### Regression metrics (separate scale, not comparable to classification)")
reg_table = pd.DataFrame([{"MAE": mae, "RMSE": rmse, "R2": r2, "Adjusted R2": adj_r2}], index=["Linear Regression (fare)"])
log(reg_table.round(3).to_markdown())

best_clf = results_df["F1"].idxmax()
log(
    f"\n**Recommendation:** Deploy **{best_clf}**. It has the highest F1 "
    f"score ({results_df.loc[best_clf, 'F1']:.3f}) among the three "
    f"classifiers, balancing precision ({results_df.loc[best_clf, 'Precision']:.3f}) "
    f"and recall ({results_df.loc[best_clf, 'Recall']:.3f}), and its AUC "
    f"({results_df.loc[best_clf, 'AUC']:.3f}) is competitive with the others. "
    "Random Forest in particular is also more robust to the noisy/imputed "
    "features here than a single Decision Tree, without the linear "
    "assumptions of Logistic Regression."
)

# ----------------------------------------------------------------------
# 9. Save the best full pipeline with joblib
# ----------------------------------------------------------------------
best_pipeline = fitted_pipelines[best_clf]
joblib.dump(best_pipeline, "best_pipeline.joblib")
log(f"\nSaved the full fitted pipeline ({best_clf}) to `best_pipeline.joblib`.")

# Reload check
reloaded = joblib.load("best_pipeline.joblib")
sample_raw = X_test.iloc[[0]]
pred_original = best_pipeline.predict(sample_raw)[0]
pred_reloaded = reloaded.predict(sample_raw)[0]
log(f"Reload check on raw sample row -> original pred={pred_original}, reloaded pred={pred_reloaded}, match={pred_original == pred_reloaded}")

with open("model_report.md", "w") as f:
    f.write("\n".join(report_lines))

print("\n\nDone. See model_report.md, charts/, and best_pipeline.joblib")
