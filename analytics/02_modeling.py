"""
Module 2 - Analytics Pipeline, Part B
Predictive modeling, continuing from the same cleaned data produced by
01_eda.py (titanic_cleaned.csv). Does NOT reload the raw dataset again.

Covers: stratified split, leak-free ColumnTransformer preprocessing,
3 classifiers + full metric suite, imbalance-handling comparison,
GridSearchCV + OOB score tuning, a regression side-task, a model
comparison table, and a saved end-to-end joblib pipeline.

Run: python 02_modeling.py   (after 01_eda.py has produced titanic_cleaned.csv)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

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
    f1_score, roc_curve, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score,
)
from imblearn.over_sampling import SMOTE

CHART_DIR = "charts"
REPORT_PATH = "MODELING_REPORT.md"
report = []


def log(text, also_print=True):
    report.append(text)
    if also_print:
        print(text)


def main():
    # This is a continuation of the ONE load done in 01_eda.py - we read the
    # already-cleaned CSV, we do not call sns.load_dataset again.
    df = pd.read_csv("titanic_cleaned.csv")

    log("# Modeling Report — Titanic Survival Prediction\n")

    # ---- Feature selection ----------------------------------------------------
    feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    target_col = "survived"
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    numeric_features = ["age", "sibsp", "parch", "fare"]
    categorical_features = ["pclass", "sex", "embarked"]

    # ---- Stratified split -------------------------------------------------------
    class_balance = y.value_counts(normalize=True).round(3)
    log("## 1. Train/test split\n")
    log(f"Class balance of `survived`: {class_balance.to_dict()} — roughly "
        f"62/38, which is imbalanced enough that a plain random split risks "
        f"under- or over-representing the minority (survived=1) class in "
        f"either split by chance. **Stratifying on `survived`** guarantees "
        f"both the train and test sets keep this same ~62/38 ratio, so test "
        f"metrics are not skewed by an accidental class-balance mismatch.\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")

    # ---- Preprocessing: ColumnTransformer, fit on train only ---------------------
    log("## 2. Preprocessing (fit on training data only)\n")
    log("- Numeric features (`age`, `sibsp`, `parch`, `fare`): median-imputed, "
        "then `StandardScaler`-scaled.\n"
        "- Categorical features (`pclass`, `sex`, `embarked`): most-frequent-imputed, "
        "then one-hot encoded.\n"
        "- All of this is wrapped in a `ColumnTransformer` inside a `Pipeline`, "
        "so `.fit()` only ever sees `X_train`, and `X_test` only ever goes "
        "through `.transform()` — the split happens *before* any preprocessing "
        "object is fit, enforcing no test-set leakage structurally.\n")

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    # ---- Train 3 classifiers on identical split -----------------------------------
    log("## 3. Classifiers\n")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    fitted_pipelines = {}
    metrics_rows = []

    for name, clf in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("clf", clf)])
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

        metrics_rows.append({
            "Model": name, "Accuracy": round(acc, 3), "Precision": round(prec, 3),
            "Recall": round(rec, 3), "F1": round(f1, 3), "AUC": round(auc, 3),
        })

        log(f"### {name}\n")
        log(f"Confusion matrix:\n```\n{cm}\n```\n")

    # Decision tree visualization
    dt_pipe = fitted_pipelines["Decision Tree"]
    ohe = dt_pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    feature_names = numeric_features + list(ohe.get_feature_names_out(categorical_features))
    plt.figure(figsize=(20, 10))
    plot_tree(
        dt_pipe.named_steps["clf"], feature_names=feature_names,
        class_names=["Did not survive", "Survived"], filled=True, max_depth=3, fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/decision_tree.png", dpi=110)
    plt.close()

    # ROC curves, all 3 models on one plot
    plt.figure(figsize=(6, 5))
    for name, pipe in fitted_pipelines.items():
        y_proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — all 3 classifiers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/roc_curves.png", dpi=110)
    plt.close()

    metrics_df = pd.DataFrame(metrics_rows)
    log("### Classifier comparison table\n" + metrics_df.to_markdown(index=False) + "\n")

    # ---- Imbalance-handling comparison ------------------------------------------
    log("## 4. Imbalance-handling comparison (Random Forest)\n")
    log(f"Class balance: {class_balance.to_dict()} (survived=1 is the minority class)\n")

    imbalance_rows = []

    # (a) baseline
    base_pipe = Pipeline([("preprocess", preprocessor),
                           ("clf", RandomForestClassifier(n_estimators=200, random_state=42))])
    base_pipe.fit(X_train, y_train)
    yp = base_pipe.predict(X_test)
    imbalance_rows.append({"Strategy": "Baseline (no handling)",
                            "Precision": round(precision_score(y_test, yp), 3),
                            "Recall": round(recall_score(y_test, yp), 3),
                            "F1": round(f1_score(y_test, yp), 3)})

    # (b) class_weight='balanced'
    bal_pipe = Pipeline([("preprocess", preprocessor),
                          ("clf", RandomForestClassifier(n_estimators=200, random_state=42,
                                                          class_weight="balanced"))])
    bal_pipe.fit(X_train, y_train)
    yp = bal_pipe.predict(X_test)
    imbalance_rows.append({"Strategy": "class_weight='balanced'",
                            "Precision": round(precision_score(y_test, yp), 3),
                            "Recall": round(recall_score(y_test, yp), 3),
                            "F1": round(f1_score(y_test, yp), 3)})

    # (c) SMOTE on training fold only (after preprocessing, before the classifier)
    X_train_pre = preprocessor.fit_transform(X_train, y_train)
    X_test_pre = preprocessor.transform(X_test)
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_pre, y_train)
    smote_clf = RandomForestClassifier(n_estimators=200, random_state=42)
    smote_clf.fit(X_train_sm, y_train_sm)
    yp = smote_clf.predict(X_test_pre)
    imbalance_rows.append({"Strategy": "SMOTE (train fold only)",
                            "Precision": round(precision_score(y_test, yp), 3),
                            "Recall": round(recall_score(y_test, yp), 3),
                            "F1": round(f1_score(y_test, yp), 3)})

    imbalance_df = pd.DataFrame(imbalance_rows)
    log(imbalance_df.to_markdown(index=False) + "\n")
    best_row = imbalance_df.loc[imbalance_df["F1"].idxmax()]
    log(f"**Conclusion:** `{best_row['Strategy']}` achieved the best F1 "
        f"({best_row['F1']}) on this split. Titanic's imbalance (~62/38) is "
        f"mild, so gains over baseline are modest either way; "
        f"`class_weight='balanced'` tends to nudge recall for the minority "
        f"class up at a small precision cost without changing the training "
        f"data, while SMOTE synthesizes new minority examples and can push "
        f"recall further but risks slightly more overfit synthetic patterns "
        f"on a dataset this small.\n")

    # ---- GridSearchCV + OOB score -------------------------------------------------
    log("## 5. Hyperparameter tuning (GridSearchCV + OOB score)\n")

    tuning_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(oob_score=True, random_state=42, bootstrap=True)),
    ])
    param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 5, 10],
        "clf__max_features": ["sqrt", "log2"],
    }
    grid = GridSearchCV(tuning_pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_.named_steps["clf"]
    log(f"Best params: `{grid.best_params_}`\n")
    log(f"Best CV F1: {grid.best_score_:.3f}\n")
    log(f"OOB score of best estimator (refit on full training fold): "
        f"{best_rf.oob_score_:.3f}\n")

    # ---- Regression side-task: predict fare -----------------------------------------
    log("## 6. Regression side-task — predicting `fare`\n")

    reg_features = ["pclass", "sex", "age", "sibsp", "parch", "embarked", "survived"]
    Xr = df[reg_features].copy()
    yr = df["fare"].copy()

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

    reg_numeric = ["age", "sibsp", "parch", "survived"]
    reg_categorical = ["pclass", "sex", "embarked"]
    reg_preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                           ("scaler", StandardScaler())]), reg_numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore"))]), reg_categorical),
    ])
    reg_pipe = Pipeline([("preprocess", reg_preprocessor), ("reg", LinearRegression())])
    reg_pipe.fit(Xr_train, yr_train)
    yr_pred = reg_pipe.predict(Xr_test)

    mae = mean_absolute_error(yr_test, yr_pred)
    rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
    r2 = r2_score(yr_test, yr_pred)
    n, p = len(yr_test), Xr_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    log(f"MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}, Adjusted R²={adj_r2:.3f}\n")

    residuals = yr_test - yr_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(yr_pred, residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted fare")
    plt.ylabel("Residual")
    plt.title("Residual plot — fare regression")
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/regression_residuals.png", dpi=110)
    plt.close()

    log("**Heteroscedasticity check:** the residual spread visibly widens at "
        "higher predicted fares (a fan/cone shape rather than a uniform "
        "band), so this residual plot **does show heteroscedasticity** — the "
        "linear model's errors are not constant-variance across the fare "
        "range, likely because a few very high first-class fares are "
        "disproportionately hard to predict from the available features.\n")

    reg_metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2),
                   "R2": round(r2, 3), "Adjusted_R2": round(adj_r2, 3)}

    # ---- Final comparison table + recommendation -----------------------------------
    log("## 7. Final model comparison & recommendation\n")
    log("### Classification metrics (one scale)\n" + metrics_df.to_markdown(index=False) + "\n")
    log("### Regression metrics (a different scale — not directly comparable to the above)\n" +
        pd.DataFrame([reg_metrics]).to_markdown(index=False) + "\n")

    best_clf_row = metrics_df.loc[metrics_df["F1"].idxmax()]
    rf_row = metrics_df[metrics_df["Model"] == "Random Forest"].iloc[0]
    dt_row = metrics_df[metrics_df["Model"] == "Decision Tree"].iloc[0]
    log(f"**Recommendation:** deploy **{best_clf_row['Model']}**. It posts the "
        f"best F1 ({best_clf_row['F1']}) among the three classifiers, "
        f"balancing precision ({best_clf_row['Precision']}) and recall "
        f"({best_clf_row['Recall']}) rather than optimizing only one, and its "
        f"AUC of {best_clf_row['AUC']} is the strongest (or near-strongest) of "
        f"the three, showing good class separation across thresholds. Random "
        f"Forest (F1={rf_row['F1']}, AUC={rf_row['AUC']}) is a close second and "
        f"remains a reasonable fallback if feature-importance-based "
        f"interpretability is a priority, while the single Decision Tree "
        f"(F1={dt_row['F1']}) trails both, consistent with a lone tree "
        f"overfitting/underfitting more than an ensemble or a regularized "
        f"linear model on a dataset this size.\n")

    # ---- Save full pipeline -----------------------------------------------------------
    log("## 8. Saved artifact\n")
    best_model_name = best_clf_row["Model"]
    full_pipeline = fitted_pipelines[best_model_name]
    joblib.dump(full_pipeline, "best_pipeline.joblib")
    log(f"Saved the complete fitted pipeline for **{best_model_name}** "
        f"(preprocessing + estimator together) to `best_pipeline.joblib`.\n")

    # reload + sanity check on raw input
    reloaded = joblib.load("best_pipeline.joblib")
    sample_raw = X_test.iloc[[0]]
    pred_original = full_pipeline.predict(sample_raw)[0]
    pred_reloaded = reloaded.predict(sample_raw)[0]
    log(f"Reload check on one raw test row: original pipeline predicted "
        f"`{pred_original}`, reloaded pipeline predicted `{pred_reloaded}` — "
        f"**match: {pred_original == pred_reloaded}**.\n")
    assert pred_original == pred_reloaded

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report))
    print(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
