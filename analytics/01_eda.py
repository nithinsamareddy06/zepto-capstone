"""
Module 2, Part A - Profiling, Cleaning, and the Data Story
================================================================
Loads the Titanic dataset ONCE via seaborn, profiles it, cleans it, saves
titanic.csv as the committed offline fallback, and produces the full EDA:
univariate analysis, bivariate analysis, a correlation heatmap, a 4-chart
"data story", and an exploratory z-score standardization check.

All written interpretations required by the brief are printed to stdout
AND collected into analytics/eda_report.md so they live as reviewable
Markdown text, per the submission rules.

Run: python3 01_eda.py
Outputs: titanic.csv, charts/*.png, eda_report.md
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

os.makedirs("charts", exist_ok=True)
report_lines = []


def log(msg=""):
    """Print to stdout and collect into the markdown report."""
    print(msg)
    report_lines.append(msg)


# ----------------------------------------------------------------------
# 1. Load ONCE, profile, save the offline fallback CSV
# ----------------------------------------------------------------------
log("# Analytics Module - EDA Report\n")
log("## 1. Load & Profile\n")

df = sns.load_dataset("titanic")

# This is the one and only load of the raw dataset for the whole module.
df.to_csv("titanic.csv", index=False)
log("Saved raw load to `titanic.csv` (committed offline fallback; "
    "02_modeling.py reads this same file and never reloads from seaborn).\n")

buf_shape = f"Shape: {df.shape[0]} rows x {df.shape[1]} columns"
log(f"- {buf_shape}")

log("\n### df.info()")
import io
info_buf = io.StringIO()
df.info(buf=info_buf)
log("```\n" + info_buf.getvalue() + "```")

log("\n### df.describe()")
log(df.describe().to_markdown())

log("\n### Missing values (%) — columns with any missing data")
missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]
log(missing_pct.round(2).to_markdown())

# ----------------------------------------------------------------------
# 2. Missing-value handling per the threshold rule
#    <5% missing  -> drop rows
#    5-30% missing -> impute
#    very high (>=~70%) -> drop column or encode "missing" as its own cat.
# ----------------------------------------------------------------------
log("\n## 2. Missing-Value Handling (threshold rule)\n")

df_clean = df.copy()
for col, pct in missing_pct.items():
    if pct < 5:
        n_before = len(df_clean)
        df_clean = df_clean.dropna(subset=[col])
        log(f"- **{col}** ({pct:.2f}% missing, <5%): dropped {n_before - len(df_clean)} rows.")
    elif pct <= 30:
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_val = df[col].median()
            df_clean[col] = df_clean[col].fillna(fill_val)
            log(f"- **{col}** ({pct:.2f}% missing, 5-30%): imputed with median = {fill_val:.2f}.")
        else:
            fill_val = df[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(fill_val)
            log(f"- **{col}** ({pct:.2f}% missing, 5-30%): imputed with mode = '{fill_val}'.")
    else:
        # High-missing column (deck is ~77% missing in this dataset).
        # Imputing 3 out of 4 values would be unreliable / mostly fabricated,
        # so we encode "missing" as its own explicit category instead of
        # dropping the column outright, preserving the (weak) signal that a
        # deck being *recorded at all* correlates with cabin class/survival.
        df_clean[col] = df_clean[col].astype(object).fillna("Missing")
        log(f"- **{col}** ({pct:.2f}% missing, very high): imputation would be unreliable, "
            f"so 'Missing' is encoded as its own explicit category rather than dropped or imputed.")

log(f"\nShape after missing-value handling: {df_clean.shape}")

# ----------------------------------------------------------------------
# 3. Univariate analysis: age & fare
# ----------------------------------------------------------------------
log("\n## 3. Univariate Analysis — age & fare\n")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
sns.histplot(df_clean["age"], kde=True, ax=axes[0, 0]).set_title("Age — Histogram")
sns.boxplot(x=df_clean["age"], ax=axes[0, 1]).set_title("Age — Box plot")
sns.histplot(df_clean["fare"], kde=True, ax=axes[1, 0]).set_title("Fare — Histogram")
sns.boxplot(x=df_clean["fare"], ax=axes[1, 1]).set_title("Fare — Box plot")
plt.tight_layout()
plt.savefig("charts/univariate_age_fare.png", dpi=120)
plt.close()
log("Saved `charts/univariate_age_fare.png`.")


def iqr_outlier_count(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum()), lower, upper


age_out, age_lo, age_hi = iqr_outlier_count(df_clean["age"])
fare_out, fare_lo, fare_hi = iqr_outlier_count(df_clean["fare"])
log(f"- IQR outliers in **age**: {age_out} (bounds: [{age_lo:.2f}, {age_hi:.2f}])")
log(f"- IQR outliers in **fare**: {fare_out} (bounds: [{fare_lo:.2f}, {fare_hi:.2f}])")

fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode()[0]
log(f"\nFare — mean={fare_mean:.2f}, median={fare_median:.2f}, mode={fare_mode:.2f}")
if fare_mean > fare_median > fare_mode:
    skew_note = ("Since mean > median > mode, the fare distribution is "
                  "**right-skewed**: a small number of high-fare (first-class) "
                  "passengers pull the mean well above the typical (median) fare.")
elif fare_mean < fare_median < fare_mode:
    skew_note = "Since mean < median < mode, the fare distribution is **left-skewed**."
else:
    skew_note = "Mean/median/mode are close together, suggesting a roughly **symmetric** distribution."
log(skew_note)

# ----------------------------------------------------------------------
# 4. Bivariate analysis via boolean masking + correlation heatmap
# ----------------------------------------------------------------------
log("\n## 4. Bivariate Analysis\n")

survival_by_sex = {
    s: df_clean.loc[df_clean["sex"] == s, "survived"].mean()
    for s in df_clean["sex"].unique()
}
log("### Survival rate by sex")
for s, rate in survival_by_sex.items():
    log(f"- {s}: {rate:.3f}")

survival_by_class = {
    c: df_clean.loc[df_clean["pclass"] == c, "survived"].mean()
    for c in sorted(df_clean["pclass"].unique())
}
log("\n### Survival rate by pclass")
for c, rate in survival_by_class.items():
    log(f"- class {c}: {rate:.3f}")

log("\n### Survival rate by sex AND pclass")
for s in df_clean["sex"].unique():
    for c in sorted(df_clean["pclass"].unique()):
        mask = (df_clean["sex"] == s) & (df_clean["pclass"] == c)
        rate = df_clean.loc[mask, "survived"].mean()
        log(f"- sex={s}, pclass={c}: {rate:.3f}")

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_matrix = df_clean[corr_cols].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (6 numeric columns)")
plt.tight_layout()
plt.savefig("charts/correlation_heatmap.png", dpi=120)
plt.close()
log("\nSaved `charts/correlation_heatmap.png`.")

# Find top-2 strongest off-diagonal absolute correlations
pairs = []
for i, c1 in enumerate(corr_cols):
    for c2 in corr_cols[i + 1:]:
        pairs.append((c1, c2, corr_matrix.loc[c1, c2]))
pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
top2 = pairs_sorted[:2]
log("\n### Two strongest correlations")
for c1, c2, val in top2:
    log(f"- **{c1} vs {c2}**: r = {val:.3f}")
def describe_pair(c1, c2, val):
    direction = "negative" if val < 0 else "positive"
    if {c1, c2} == {"pclass", "fare"}:
        note = ("this is expected: pclass is coded 1 (best) to 3 (worst), so "
                "a negative correlation with fare simply means higher-class "
                "tickets cost more.")
    elif {c1, c2} == {"sibsp", "parch"}:
        note = ("this reflects that passengers traveling with siblings/spouses "
                "also tend to be traveling with parents/children, i.e. it "
                "picks out passengers traveling as part of a larger family unit.")
    elif "survived" in (c1, c2):
        other = c2 if c1 == "survived" else c1
        note = (f"this indicates {other} has a meaningfully {direction} "
                 f"relationship with survival outcome.")
    else:
        note = f"this reflects a {direction} relationship between the two features."
    return note


log("\nInterpretation:")
for c1, c2, val in top2:
    log(f"- **{c1} vs {c2}** (r={val:.2f}): {describe_pair(c1, c2, val)}")

# ----------------------------------------------------------------------
# 5. Multivariate "data story" — at least 4 charts, each with a written note
# ----------------------------------------------------------------------
log("\n## 5. Multivariate Data Story (4+ charts)\n")

# Chart 1: survival rate by class and sex (bar)
plt.figure(figsize=(6, 4))
sns.barplot(data=df_clean, x="pclass", y="survived", hue="sex")
plt.title("Survival rate by class and sex")
plt.tight_layout()
plt.savefig("charts/story_1_survival_class_sex.png", dpi=120)
plt.close()
log("**Chart 1 — Survival rate by class & sex (bar):** Women survived at far "
    "higher rates than men in every class, and within each sex, survival "
    "drops sharply from 1st to 3rd class. This suggests both the 'women and "
    "children first' evacuation norm and unequal access to lifeboats by "
    "cabin class were both at play, rather than either factor alone "
    "explaining the outcome.")

# Chart 2: age distribution by survival (box)
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_clean, x="survived", y="age")
plt.title("Age distribution by survival")
plt.tight_layout()
plt.savefig("charts/story_2_age_by_survival.png", dpi=120)
plt.close()
log("\n**Chart 2 — Age by survival (box):** Median age is broadly similar "
    "between survivors and non-survivors, but survivors show a visibly "
    "wider lower tail, consistent with young children being prioritized "
    "for rescue even though age alone is a weak predictor overall.")

# Chart 3: fare vs age scatter, colored by survival
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df_clean, x="age", y="fare", hue="survived", alpha=0.6)
plt.title("Fare vs Age, colored by survival")
plt.tight_layout()
plt.savefig("charts/story_3_fare_age_survival.png", dpi=120)
plt.close()
log("\n**Chart 3 — Fare vs age, colored by survival (scatter):** "
    "Survivors are visibly concentrated at higher fares, while the cluster "
    "of low-fare passengers across all ages skews toward non-survival. "
    "Fare, as a proxy for cabin class and deck location, looks like a "
    "stronger survival signal than age by itself.")

# Chart 4: family size vs survival (bar), engineered feature
df_clean["family_size"] = df_clean["sibsp"] + df_clean["parch"]
plt.figure(figsize=(6, 4))
sns.barplot(data=df_clean, x="family_size", y="survived")
plt.title("Survival rate by family size (sibsp + parch)")
plt.tight_layout()
plt.savefig("charts/story_4_family_size.png", dpi=120)
plt.close()
log("\n**Chart 4 — Survival rate by family size (bar):** Passengers "
    "traveling with a small family (1-3 relatives aboard) survived at "
    "higher rates than those traveling completely alone or in very large "
    "families, suggesting mutual help in small groups, while large "
    "families struggled to stay together and evacuate as a unit.")

# ----------------------------------------------------------------------
# 6. Exploratory z-score standardization check (EDA-stage only)
# ----------------------------------------------------------------------
log("\n## 6. Exploratory Standardization Check (age & fare)\n")

for col in ["age", "fare"]:
    mean_before, std_before = df_clean[col].mean(), df_clean[col].std()
    z = (df_clean[col] - mean_before) / std_before
    log(f"- **{col}** — before: mean={mean_before:.3f}, std={std_before:.3f} | "
        f"after z-score: mean={z.mean():.3f}, std={z.std():.3f}")

log(
    "\nNote: this standardization is an EDA-stage sanity check only. It does "
    "not feed into the modeling pipeline in `02_modeling.py`, which performs "
    "its own train-only scaling to avoid leakage."
)

# ----------------------------------------------------------------------
# Save report
# ----------------------------------------------------------------------
with open("eda_report.md", "w") as f:
    f.write("\n".join(report_lines))

print("\n\nDone. See eda_report.md and charts/ for all outputs.")
