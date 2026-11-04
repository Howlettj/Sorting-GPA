"""
Exploratory Data Analysis Pipeline — Generic Version
------------------------------------------------------
Assumes you know NOTHING about the dataset except which column is
the target. Everything else (numeric vs. categorical columns, which
features to plot, etc.) is detected automatically from the data.

Only two things to set below: the file path and the target column name.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------------------------------------------
# 0. Config — the only two things you need to know
# -----------------------------------------------------------------
FILE_PATH = "your_file.csv"
TARGET_COL = "target"   # the column you're trying to predict

sns.set_style("whitegrid")

# -----------------------------------------------------------------
# 1. Load and do a first-pass inspection
# -----------------------------------------------------------------
df = pd.read_csv(FILE_PATH)

print("Shape:", df.shape)
print("\nInfo:")
df.info()
print("\nHead:")
print(df.head())
print("\nDtypes:\n", df.dtypes)

assert TARGET_COL in df.columns, f"'{TARGET_COL}' not found in columns: {list(df.columns)}"

# -----------------------------------------------------------------
# 2. Auto-detect column types
#    (treats low-cardinality numeric columns as categorical too,
#    since things like binary flags or small codes often get read
#    in as ints/floats)
# -----------------------------------------------------------------
CARDINALITY_THRESHOLD = 15  # numeric cols with <= this many unique values are treated as categorical

raw_numeric = df.select_dtypes(include=np.number).columns.tolist()
raw_categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

low_card_numeric = [c for c in raw_numeric if df[c].nunique() <= CARDINALITY_THRESHOLD]

numeric_cols = [c for c in raw_numeric if c not in low_card_numeric]
categorical_cols = raw_categorical + low_card_numeric

# Target is its own thing — pull it out of whichever list it landed in
target_is_numeric = TARGET_COL in raw_numeric and TARGET_COL not in low_card_numeric
numeric_cols = [c for c in numeric_cols if c != TARGET_COL]
categorical_cols = [c for c in categorical_cols if c != TARGET_COL]

print(f"\nTarget '{TARGET_COL}' detected as: {'numeric' if target_is_numeric else 'categorical'}")
print(f"Numeric feature columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical feature columns ({len(categorical_cols)}): {categorical_cols}")

# -----------------------------------------------------------------
# 3. Summary statistics
# -----------------------------------------------------------------
if numeric_cols:
    print("\n--- Numeric summary ---")
    print(df[numeric_cols + [TARGET_COL] if target_is_numeric else numeric_cols].describe())

    skew_check = pd.DataFrame({
        "mean": df[numeric_cols].mean(),
        "median": df[numeric_cols].median(),
        "skew": df[numeric_cols].skew()
    }).sort_values("skew", ascending=False)
    print("\nMean vs Median (skew indicator, sorted by skew):\n", skew_check)

if categorical_cols:
    print("\n--- Categorical summary ---")
    print(df[categorical_cols].describe())
    for col in categorical_cols:
        print(f"\nValue counts for '{col}':")
        print(df[col].value_counts(dropna=False).head(10))

# -----------------------------------------------------------------
# 4. Missing values
# -----------------------------------------------------------------
missing_count = df.isnull().sum()
missing_pct = df.isnull().mean() * 100
missing_summary = pd.DataFrame({"count": missing_count, "pct": missing_pct})
missing_summary = missing_summary[missing_summary["count"] > 0].sort_values(
    "pct", ascending=False
)
print("\nMissing values:\n", missing_summary if not missing_summary.empty else "None found.")

if missing_summary.shape[0] > 0:
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title("Missing Value Pattern")
    plt.tight_layout()
    plt.savefig("missing_values_heatmap.png")
    plt.close()

# -----------------------------------------------------------------
# 5. Duplicates
# -----------------------------------------------------------------
full_dupes = df.duplicated().sum()
print(f"\nFull-row duplicates: {full_dupes}")

# Try to guess an ID-like column (unique per row, name suggests identifier)
id_like_cols = [c for c in df.columns if df[c].nunique() == len(df) and
                ("id" in c.lower() or "key" in c.lower())]
if id_like_cols:
    print(f"Likely ID column(s) detected: {id_like_cols} "
          f"(these are unique per row, so no duplicate check needed there)")

# -----------------------------------------------------------------
# 6. Outlier detection (IQR method — robust to skew)
#    Runs on every numeric feature automatically
# -----------------------------------------------------------------
def iqr_outliers(series):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    mask = (series < lower) | (series > upper)
    return mask, lower, upper

outlier_summary = {}
for col in numeric_cols:
    mask, lower, upper = iqr_outliers(df[col])
    outlier_summary[col] = {
        "n_outliers": mask.sum(),
        "pct_outliers": round(mask.mean() * 100, 2),
        "lower_bound": lower,
        "upper_bound": upper,
    }

if outlier_summary:
    outlier_df = pd.DataFrame(outlier_summary).T.sort_values(
        "pct_outliers", ascending=False
    )
    print("\nOutlier summary (IQR method):\n", outlier_df)

    n_cols_grid = 3
    n_rows_grid = int(np.ceil(len(numeric_cols) / n_cols_grid))
    fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(5 * n_cols_grid, 4 * n_rows_grid))
    axes = np.array(axes).flatten()
    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig("outlier_boxplots.png")
    plt.close()

# -----------------------------------------------------------------
# 7. Distribution plots — every numeric feature split by every
#    categorical feature (only makes plots if both types exist)
# -----------------------------------------------------------------
MAX_CATEGORIES_TO_PLOT = 10  # skip categorical cols with too many unique values to plot cleanly

for num_col in numeric_cols:
    for cat_col in categorical_cols:
        if df[cat_col].nunique() > MAX_CATEGORIES_TO_PLOT:
            continue
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=cat_col, y=num_col, data=df)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{num_col} by {cat_col}")
        plt.tight_layout()
        plt.savefig(f"dist_{num_col}_by_{cat_col}.png")
        plt.close()

print("\nDistribution plots saved for each numeric x categorical pair "
      "(categorical columns with > {} unique values were skipped).".format(MAX_CATEGORIES_TO_PLOT))

# -----------------------------------------------------------------
# 8. Correlation with the target
# -----------------------------------------------------------------
if target_is_numeric:
    corr_cols = numeric_cols + [TARGET_COL]
    corr_matrix = df[corr_cols].corr()
    target_corr = corr_matrix[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)
    print(f"\nCorrelation of numeric features with '{TARGET_COL}' "
          f"(sorted by strength):\n", target_corr)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()

    # Categorical features vs numeric target: eta-squared (ANOVA effect size)
    def eta_squared(df, cat_col, num_col):
        groups = [g[num_col].dropna().values for _, g in df.groupby(cat_col) if len(g) > 1]
        if len(groups) < 2:
            return np.nan, np.nan, np.nan
        f_val, p_val = stats.f_oneway(*groups)
        grand_mean = df[num_col].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = sum((df[num_col].dropna() - grand_mean) ** 2)
        eta_sq = ss_between / ss_total
        return eta_sq, f_val, p_val

    print(f"\nCategorical feature association with '{TARGET_COL}' (ANOVA eta-squared):")
    for col in categorical_cols:
        eta_sq, f_val, p_val = eta_squared(df, col, TARGET_COL)
        print(f"  {col}: eta^2={eta_sq:.3f}, F={f_val:.2f}, p={p_val:.4f}")

else:
    # Categorical target: numeric features via point-biserial / group means,
    # categorical features via chi-square
    print(f"\nNumeric feature means grouped by target '{TARGET_COL}':")
    print(df.groupby(TARGET_COL)[numeric_cols].mean())

    print(f"\nChi-square test: categorical features vs target '{TARGET_COL}':")
    for col in categorical_cols:
        contingency = pd.crosstab(df[col], df[TARGET_COL])
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
        print(f"  {col}: chi2={chi2:.2f}, p={p_val:.4f}")

    # Correlation heatmap for numeric features only (target excluded, it's categorical)
    if numeric_cols:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap (numeric features only)")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png")
        plt.close()

print("\nEDA complete. Plots saved as PNG files in the working directory.")
