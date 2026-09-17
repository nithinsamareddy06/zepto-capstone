# Module 2 — Analytics Pipeline (`/analytics`)

One cohesive pipeline on the Titanic dataset (`sns.load_dataset('titanic')`,
loaded exactly once): profile it, clean it, tell a visual survival story
about it, then build and rigorously evaluate a full predictive-modeling
pipeline on the same cleaned data.

## Files

| File | Purpose |
|---|---|
| `01_eda.py` | Loads the dataset once, saves `titanic.csv` (offline fallback) and `titanic_cleaned.csv`, profiles, cleans, and produces the full EDA story. Findings written to `EDA_FINDINGS.md`. |
| `02_modeling.py` | Reads `titanic_cleaned.csv` (no second load), runs the full classification + regression pipeline. Report written to `MODELING_REPORT.md`. |
| `titanic.csv` | Offline fallback of the *raw* loaded dataset — `pd.read_csv("titanic.csv")` works with no network. |
| `titanic_cleaned.csv` | Cleaned dataset that `02_modeling.py` builds on. |
| `charts/` | All saved chart PNGs (histograms, boxplots, heatmap, story charts, decision tree, ROC curves, residual plot). |
| `best_pipeline.joblib` | The complete fitted preprocessing+estimator pipeline (see Task 8), reloadable with `joblib.load`. |
| `EDA_FINDINGS.md`, `MODELING_REPORT.md` | All required written interpretations, tables, and conclusions. |

## How to run

```bash
pip install -r requirements.txt
python 01_eda.py        # profiling, cleaning, EDA story, standardization check
python 02_modeling.py   # classifiers, imbalance comparison, tuning, regression, saved pipeline
```

`01_eda.py` needs internet on its *first* run only, to fetch and cache the
Titanic dataset from Seaborn's data repository; every later run reuses the
cache (or `titanic.csv`) and needs no network.

## Design decisions & summary of results

**Missing-value strategy.** Applied per-column using the assignment's
threshold rule: `<5%` missing → drop those rows; `5–30%` → impute
(median for numeric, mode for categorical); very-high missing (`deck`,
~77%) → too unreliable to impute meaningfully, so it's encoded as its own
`"Unknown"` category rather than dropped outright, since "cabin unknown"
itself carries signal correlated with class. Exact percentages per column
are logged in `EDA_FINDINGS.md` §2.

**Outliers & skew.** IQR-rule outlier counts are reported for both `age`
and `fare` in `EDA_FINDINGS.md` §3. `fare`'s mean > median > mode ordering
confirms a right-skewed distribution, driven by a small number of very
high first-class fares.

**Correlation.** The 6×6 correlation matrix is restricted to `survived,
pclass, age, sibsp, parch, fare` — `adult_male` and `alone` are excluded as
derived/redundant flags. The two strongest off-diagonal pairs
(`pclass`↔`fare` and `sibsp`↔`parch`) are interpreted in `EDA_FINDINGS.md` §4.

**EDA-stage standardization** of `age`/`fare` (§6) is a pure sanity check
and is **not** reused by the modeling pipeline — Part B fits its own
train-only `StandardScaler` inside a `ColumnTransformer`, exactly as
required, to avoid any leakage from the full/EDA-stage transform.

**Modeling pipeline.** Stratified 80/20 split on `survived` (justified by
the ~62/38 class balance). All preprocessing (imputation, one-hot encoding,
scaling) lives inside a `ColumnTransformer` + `Pipeline`, fit only on the
training fold. Three classifiers (Logistic Regression, Decision Tree,
Random Forest) are trained and evaluated with the full metric suite
(confusion matrix, accuracy, precision, recall, F1, ROC/AUC) — see
`MODELING_REPORT.md` §3 for the comparison table and `charts/roc_curves.png`
/ `charts/decision_tree.png`.

**Imbalance handling.** Baseline vs. `class_weight='balanced'` vs. SMOTE
(applied to the training fold only, after the split, to avoid leakage) are
compared on Random Forest in §4, with a written conclusion on which
performed best on this split.

**Tuning.** `GridSearchCV` over `n_estimators`, `max_depth`, `max_features`
for `RandomForestClassifier(oob_score=True, ...)`; best params and OOB
score reported in §5.

**Regression side-task.** A separate `LinearRegression` predicts `fare`
from the other features; MAE/RMSE/R²/Adjusted R² and a residual plot are
in §6, with an explicit heteroscedasticity conclusion (the residual spread
widens at higher predicted fares — heteroscedastic).

**Final comparison & recommendation** (§7) presents classification metrics
and regression metrics as two separate metric groups (different scales,
not merged), with a written recommendation of which classifier to deploy,
referencing specific metric values.

**Saved artifact.** The best-performing classifier's *complete* fitted
pipeline (preprocessing + estimator together) is saved via
`joblib.dump(full_pipeline, "best_pipeline.joblib")`, then reloaded and
verified to predict identically on a raw, unpreprocessed test row (§8).

> Exact numbers (accuracy/F1/AUC/etc.) depend on the random seed and are
> recorded verbatim in `MODELING_REPORT.md` rather than restated here, so
> this README doesn't drift out of sync with the actual run.
