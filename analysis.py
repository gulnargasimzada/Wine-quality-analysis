import time
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from the specified CSV file path."""
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame):
    """Clean data, separate features and target, and split into train/test sets."""
    if df.empty:
        raise ValueError("DataFrame cannot be empty.")

    numeric_df = df.select_dtypes(include=["float64", "int64"]).dropna()
    if "quality" not in numeric_df.columns:
        raise KeyError("'quality' column is missing from the dataset.")

    X = numeric_df.drop(columns=["quality"])
    y = numeric_df["quality"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train) -> LinearRegression:
    """Fit a Linear Regression model on training features and targets."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LinearRegression, X_test, y_test):
    """Generate predictions and compute regression evaluation metrics."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


def plot_quality_distribution(df: pd.DataFrame, save_path: str = None):
    """Plot distribution of wine quality scores."""
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="quality", bins=range(3, 11), kde=True)
    plt.title("Distribution of Wine Quality Scores")
    plt.xlabel("Quality Score")
    plt.ylabel("Count")
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_alcohol_vs_quality(df: pd.DataFrame, save_path: str = None):
    """Scatter plot of alcohol content vs quality score."""
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="alcohol", y="quality", hue="type", alpha=0.5)
    plt.title("Alcohol Content vs Quality Score")
    plt.xlabel("Alcohol (%)")
    plt.ylabel("Quality Score")
    if save_path:
        plt.savefig(save_path)
    plt.close()


def benchmark_pandas(file_path: str):
    """Benchmark Pandas filtering and aggregation operations."""
    start_time = time.perf_counter()
    df = pd.read_csv(file_path)
    filtered_df = df[(df["quality"] >= 6) & (df["alcohol"] > 10.0)]
    agg_df = (
        filtered_df.groupby("quality")
        .agg({"alcohol": "mean", "fixed acidity": "mean"})
        .reset_index()
    )
    duration = time.perf_counter() - start_time
    return duration, agg_df


def benchmark_polars(file_path: str):
    """Benchmark Polars filtering and aggregation operations."""
    start_time = time.perf_counter()
    df = pl.read_csv(file_path)
    filtered_df = df.filter((pl.col("quality") >= 6) & (pl.col("alcohol") > 10.0))
    agg_df = filtered_df.group_by("quality").agg(
        [
            pl.col("alcohol").mean().alias("alcohol_mean"),
            pl.col("fixed acidity").mean().alias("fixed_acidity_mean"),
        ]
    )
    duration = time.perf_counter() - start_time
    return duration, agg_df


def run_pipeline(csv_file_path: str = "Data/wine_quality_merged.csv"):
    """Execute the full analytical, modeling, and benchmarking pipeline."""
    df = load_data(csv_file_path)

    print("--- Head (First 5 Rows) ---")
    print(df.head())
    print("\n--- Summary Statistics ---")
    print(df.describe())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)

    mse, r2 = evaluate_model(model, X_test, y_test)
    print("\nModel training complete!")
    print(f"Mean Squared Error: {mse:.3f}")
    print(f"R² Score: {r2:.3f}")

    plot_quality_distribution(df)
    plot_alcohol_vs_quality(df)

    pandas_time, _ = benchmark_pandas(csv_file_path)
    polars_time, _ = benchmark_polars(csv_file_path)
    print(f"\nPandas Execution Time: {pandas_time * 1000:.2f} ms")
    print(f"Polars Execution Time: {polars_time * 1000:.2f} ms")
    speedup = pandas_time / polars_time
    print(f"Polars is {speedup:.2f}x faster than Pandas!")

    return {
        "model": model,
        "mse": mse,
        "r2": r2,
        "speedup": speedup
    }


if __name__ == "__main__":
    run_pipeline()