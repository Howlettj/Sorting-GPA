"""Feature engineering pipeline for the job-duration prediction model (Phase 2)."""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/raw/synthetic_jobs.csv"
OUT_DIR = "data/processed"
TARGET = "actual_duration_sec"
PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["submitted_at"])
    df = df.drop_duplicates(subset="job_id").reset_index(drop=True)
    df["memory_gb"] = df.groupby("cpu_cores")["memory_gb"].transform(lambda s: s.fillna(s.median()))
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["priority_rank"] = df["priority"].map(PRIORITY_ORDER)
    df["log_size_mb"] = np.log1p(df["size_mb"])
    df["complexity_score"] = df["log_size_mb"] * (1 + df["dependencies"]) / df["cpu_cores"]
    df["resource_ratio"] = df["memory_gb"] / df["cpu_cores"]
    df["dependency_density"] = df["dependencies"] / df["log_size_mb"].replace(0, np.nan)
    df["dependency_density"] = df["dependency_density"].fillna(0)

    df["day_of_week"] = df["submitted_at"].dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["submitted_at"].dt.month
    df["hour_of_day"] = df["submitted_at"].dt.hour

    job_type_dummies = pd.get_dummies(df["job_type"], prefix="job_type", drop_first=True)
    df = pd.concat([df, job_type_dummies], axis=1)
    for col in job_type_dummies.columns:
        df[f"{col}_x_priority"] = job_type_dummies[col] * df["priority_rank"]

    return df


def select_features(df: pd.DataFrame, target: str) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target)

    variances = df[numeric_cols].var()
    near_zero_variance = variances[variances < 1e-6].index.tolist()

    corr_with_target = df[numeric_cols].corrwith(df[target]).sort_values(key=abs, ascending=False)

    f_stats, p_values = f_regression(df[numeric_cols].fillna(0), df[target])
    anova_scores = pd.Series(p_values, index=numeric_cols).sort_values()

    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(df[numeric_cols].fillna(0), df[target])
    importance = pd.Series(rf.feature_importances_, index=numeric_cols).sort_values(ascending=False)

    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    redundant_pairs = [
        (col, row, upper.loc[row, col])
        for col in upper.columns
        for row in upper.index
        if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > 0.9
    ]

    return {
        "near_zero_variance": near_zero_variance,
        "corr_with_target": corr_with_target,
        "anova_p_values": anova_scores,
        "rf_importance": importance,
        "redundant_pairs": redundant_pairs,
    }


def scale_and_split(df: pd.DataFrame, target: str, drop_cols: list[str]):
    feature_cols = [c for c in df.columns if c not in drop_cols + [target]]
    X = df[feature_cols]
    y = df[target]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    scaler_params = {
        "feature_names": feature_cols,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test), scaler_params


def main():
    df = load_and_clean(RAW_PATH)
    df = create_features(df)

    selection = select_features(df, TARGET)

    print("Near-zero-variance features:", selection["near_zero_variance"] or "none")
    print("\nTop correlations with target:\n", selection["corr_with_target"].head(10))
    print("\nTop RF feature importances:\n", selection["rf_importance"].head(10))
    print("\nLowest ANOVA p-values:\n", selection["anova_p_values"].head(10))
    print("\nRedundant feature pairs (|corr| > 0.9):", selection["redundant_pairs"] or "none")

    drop_cols = ["job_id", "submitted_at", "priority", "job_type"] + selection["near_zero_variance"]
    train, val, test, scaler_params = scale_and_split(df, TARGET, drop_cols)

    train[0].assign(**{TARGET: train[1]}).to_csv(f"{OUT_DIR}/train.csv", index=False)
    val[0].assign(**{TARGET: val[1]}).to_csv(f"{OUT_DIR}/val.csv", index=False)
    test[0].assign(**{TARGET: test[1]}).to_csv(f"{OUT_DIR}/test.csv", index=False)

    with open(f"{OUT_DIR}/scaler_params.json", "w") as f:
        json.dump(scaler_params, f, indent=2)

    print(f"\nSaved train/val/test splits and scaler_params.json to {OUT_DIR}/")
    print(f"Train: {train[0].shape}, Val: {val[0].shape}, Test: {test[0].shape}")


if __name__ == "__main__":
    main()
