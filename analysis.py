import time
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

csv_file_path = "Data/wine_quality_merged.csv"
df = pd.read_csv(csv_file_path)

print(df.head())
df.info()
print(df.describe())
print(df.isnull().sum())

numeric_df = df.select_dtypes(include=["float64", "int64"]).dropna()
X = numeric_df.drop(columns=["quality"])
y = numeric_df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

print("Model training complete!")

# making predictions on the test set
y_pred = model.predict(X_test)

# evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.3f}")
print(f"R² Score: {r2:.3f}")

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="quality", bins=range(3, 11), kde=True)
plt.title("Distribution of Wine Quality Scores")
plt.xlabel("Quality Score")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="alcohol", y="quality", hue="type", alpha=0.5)
plt.title("Alcohol Content vs Quality Score")
plt.xlabel("Alcohol (%)")
plt.ylabel("Quality Score")
plt.show()


# --- 1. Pandas Benchmark ---
def benchmark_pandas(file_path):
    start_time = time.perf_counter()

    # Read dataset
    df = pd.read_csv(file_path)

    # Filter: quality >= 6 and alcohol > 10.0
    filtered_df = df[(df["quality"] >= 6) & (df["alcohol"] > 10.0)]

    # Group by quality and calculate average alcohol and fixed acidity
    agg_df = (
        filtered_df.groupby("quality")
        .agg(
            {
                "alcohol": "mean",
                "fixed acidity": "mean",
            }
        )
        .reset_index()
    )

    duration = time.perf_counter() - start_time
    return duration, agg_df


# --- 2. Polars Benchmark ---
def benchmark_polars(file_path):
    start_time = time.perf_counter()

    # Read dataset
    df = pl.read_csv(file_path)

    # Filter: quality >= 6 and alcohol > 10.0
    filtered_df = df.filter((pl.col("quality") >= 6) & (pl.col("alcohol") > 10.0))

    # Group by quality and calculate average alcohol and fixed acidity
    agg_df = filtered_df.group_by("quality").agg(
        [
            pl.col("alcohol").mean().alias("alcohol_mean"),
            pl.col("fixed acidity").mean().alias("fixed_acidity_mean"),
        ]
    )

    duration = time.perf_counter() - start_time
    return duration, agg_df


# Execution and comparison
pandas_time, _ = benchmark_pandas(csv_file_path)
polars_time, _ = benchmark_polars(csv_file_path)

print(f"Pandas Execution Time: {pandas_time * 1000:.2f} ms")
print(f"Polars Execution Time: {polars_time * 1000:.2f} ms")

speedup = pandas_time / polars_time
print(f"Polars is {speedup:.2f}x faster than Pandas!")