# Wine Quality Analysis — Series 1 (Week 1)

## Project Overview

This project is the first part of a 3-week data engineering assignment. The goal of this series is to become comfortable with a real-world dataset by performing exploratory data analysis (EDA) using Pandas, applying basic filtering and grouping, beginning to experiment with a machine learning algorithm, and visualizing key patterns in the data.

## Dataset

**Source:** [Red and White Wine Quality Dataset (Kaggle)](https://www.kaggle.com/datasets/amirmohamadrezaie/red-and-white-wine-quality), originally derived from the **UCI Machine Learning Repository — Wine Quality Dataset** (Cortez et al., 2009), based on physicochemical tests of the Portuguese "Vinho Verde" wine.

**File used:** `wine_quality_merged.csv` — a merged dataset containing both red and white wine samples, distinguished by a `type` column.

**Why this dataset was chosen:** All input features are numeric (no text cleaning required), there are no missing values, and the target variable (`quality`) supports both regression and classification approaches — making it well suited for a first hands-on data analysis project.

### Columns

| Column | Description |
|---|---|
| fixed acidity, volatile acidity, citric acid | Acidity-related measurements |
| residual sugar, chlorides | Sweetness and salt content |
| free sulfur dioxide, total sulfur dioxide | Preservative levels |
| density, pH, sulphates, alcohol | Physical/chemical properties |
| quality | Sensory quality score (0–10), based on human tasters |
| type | Wine category: `red` or `white` |

## Steps Performed

### 1. Importing the Dataset

The dataset was loaded into a Pandas DataFrame using:
```python
df = pd.read_csv('data/wine_quality_merged.csv')
```

### 2. Inspecting the Data

- `.head()` was used to preview the first 5 rows and confirm the data loaded correctly, with all expected columns present.
- `.info()` showed that the dataset contains **6,497 rows** and **13 columns**, with all numeric columns correctly typed as `float64`/`int64` and `type` as an `object` (text) column. No missing values were found in any column.
- `.describe()` provided summary statistics (mean, std, min, max, and quartiles) for all numeric columns, giving an initial sense of the scale and spread of each feature. For example, `quality` ranges from 3 to 9, with a median around 6.

### 3. Duplicate Investigation

Running `df.duplicated().sum()` revealed **1,177 duplicate rows** (~18% of the dataset) — a substantial and unexpected proportion, which warranted deeper investigation rather than a default "drop and move on" decision.

**Investigation process:**
- Duplicate rows were isolated and visually inspected using `df[df.duplicated(keep=False)]`.
- All observed duplicate groups belonged to the **same wine type** (i.e., no red/white pairs matched each other), consistent with red and white wines having distinct chemical profiles.
- Some values appeared not just twice, but **three times**, ruling out a simple one-off data entry error.
- The dataset contains **no unique identifier column** (no `id` field), meaning duplicate rows cannot be distinguished from genuinely repeated measurements using the available columns.
- Given that each row is defined by **11 continuous physicochemical measurements**, the probability that two *entirely unrelated* wine samples would coincidentally match on all 11 values is statistically very low. A more plausible explanation is that these rows represent **multiple bottles/samples drawn from the same production batch**, which would naturally share nearly identical lab measurements.

**Decision:**
- For **EDA and visualization**, the **original dataset (with duplicates)** was retained, since these rows likely represent real, physically distinct wine samples rather than data errors — removing them would distort the true observed distribution.
- For **machine learning**, a **de-duplicated copy** (`df_model`) was created before model training. This was done specifically to prevent **data leakage**: if a duplicate row were split between the training and test sets, the model would effectively be evaluated on data it had already "seen," artificially inflating performance metrics.

### 4. Filtering and Grouping

- **Filtering:** Extracted a subset of high-quality wines using `df[df['quality'] >= 7]`, isolating wines rated 7 or above.
- **Grouping:** Used `df.groupby('type')['quality'].agg(['mean', 'count', 'min', 'max'])` to compare red and white wines. Key finding: **white wines have a slightly higher average quality score than red wines** in this dataset. The `count` also revealed a class imbalance — white wine samples (~4,898) far outnumber red wine samples (~1,599).

### 5. Exploring a Machine Learning Algorithm

**Goal chosen:** Predict the `quality` score (a numeric value) — framed as a **regression** problem.

**Algorithm chosen:** **Linear Regression**, selected for its simplicity and interpretability as a first modeling attempt.

**Steps:**
1. Created a de-duplicated modeling dataset (`df_model`) to avoid data leakage.
2. Encoded the categorical `type` column into numeric form (`red` → 0, `white` → 1) using `.map()`, since scikit-learn models require numeric input.
3. Separated the data into features (`X`, all columns except `quality`) and target (`y`, the `quality` column).
4. Split the data into training (80%) and test (20%) sets using `train_test_split`, with a fixed `random_state` to ensure reproducibility.
5. Trained a `LinearRegression` model on the training set.
6. Evaluated the model on the unseen test set.

**Results:**
- **Mean Squared Error (MSE): 0.548** — on average, the model's predictions deviate from the true quality score by roughly ±0.74 points (RMSE ≈ √0.548).
- **R² Score: 0.331** — the model explains about 33% of the variance in wine quality scores.

**Interpretation:** This is a reasonable baseline result rather than a poor one. `quality` is a **subjective human rating**, and the available features are limited to chemical measurements — factors such as grape variety, vintage, or production method (which likely also influence perceived quality) are not present in the dataset due to privacy/logistics constraints noted by the original data providers. Published research using this same dataset with linear models typically reports R² values in a similar 0.25–0.40 range, suggesting the model's performance is in line with expectations rather than indicating an implementation issue.

**Possible future improvements** (not yet implemented in this series):
- Trying non-linear models (e.g., Random Forest, Gradient Boosting) that may better capture non-linear relationships between chemical properties and taste.
- Feature engineering (e.g., ratios between related chemical properties).
- Checking for multicollinearity among features (e.g., density and alcohol are likely correlated).

### 6. Visualization

Two plots were created to visually explore patterns already identified through grouping and modeling. Both use the **original (non-deduplicated) dataset**, since the goal here was to reflect the full set of real observed samples rather than the leakage-safe modeling subset.

**Histogram — Distribution of Quality Scores**
Shows that quality scores are heavily concentrated around 5, 6, and 7, forming a roughly bell-shaped distribution. Extreme scores (3, 4, 8, 9) are rare. This explains why the regression model performs better on typical (mid-range) wines than on rare, extreme-quality wines — there is simply less data to learn from at the extremes.

**Scatter Plot — Alcohol Content vs. Quality (colored by type)**
Shows a general positive trend: wines with higher alcohol content tend to receive higher quality scores, though the relationship is noisy rather than a clean line — many wines with similar alcohol content received very different quality scores. This visually explains why alcohol is a useful predictor in the regression model, but not a sufficient one on its own (consistent with the moderate R² score). The plot also visually confirms the class imbalance observed earlier, with white wine samples (orange) clearly outnumbering red wine samples (blue).

## Key Findings Summary

1. The dataset is clean (no missing values) but contains a significant number of duplicate rows, most plausibly explained by repeated sampling from the same wine batches rather than data entry errors.
2. White wines have a slightly higher average quality score than red wines, and are represented roughly 3x more frequently in the dataset.
3. A simple Linear Regression model can explain about a third of the variance in wine quality using only chemical measurements — a reasonable starting point given the subjective nature of the target variable.
4. Alcohol content shows a visible positive association with quality, though the relationship is not strictly linear or deterministic.

---

# Part 2: Rust Ownership Experiments & Performance Benchmarking


The second component of this assignment explores foundational systems programming concepts in **Rust** using an interactive Jupyter notebook (`Rust_testing.ipynb` powered by the Evcxr kernel), followed by a performance benchmark comparing **Pandas** with **Polars** on the wine quality dataset.

---

## 7. Rust Ownership & Memory Safety Experiments

Rust achieves memory safety without relying on a garbage collector through its compile-time **Ownership** system. In `Rust_testing.ipynb`, four core memory management mechanics were demonstrated using data pipeline terminology:

### 1. Move Semantics
- In Rust, assigning an allocated heap object (like a `String`) to a new variable transfers ownership of the underlying buffer.
- Attempting to access the original variable after the move triggers a compile-time error: `[E0382] borrow of moved value`.
- This prevents **double-free** vulnerabilities at runtime by ensuring every resource has exactly one owner responsible for deallocation.

### 2. Immutable Borrowing (`&`)
- To inspect or read values across multiple functions without relinquishing ownership, Rust provides references (`&T`).
- Multiple immutable references can coexist simultaneously, allowing read-only access while guaranteeing that the underlying memory remains unmodified and valid.

### 3. Mutable Borrowing (`&mut`) & Lexical Scopes
- To mutate a value in-place, a mutable reference (`&mut T`) is required.
- Rust strictly enforces the **Aliasing XOR Mutability** rule: you may have any number of immutable references OR exactly one mutable reference at a time, but never both.
- During experimentation, attempting multiple simultaneous mutable borrows triggered `[E0499] cannot borrow as mutable more than once at a time`.
- **Solution:** Enclosed the borrows within explicit lexical scopes (`{ ... }`). Once the first mutable borrow went out of scope, its borrow was released, safely permitting subsequent mutations without memory race conditions.

### 4. Deep Copying (`.clone()`)
- When independent ownership of data is required rather than borrowed access, `.clone()` explicitly duplicates the heap allocation.
- This resolves ownership conflicts at the cost of heap allocation overhead, decoupling the lifecycle of both objects.

---

## 8. Performance Benchmark: Pandas vs. Polars

To assess modern data engineering tools, an identical data manipulation workload was executed and timed using both **Pandas** and **Polars** in Python.

### Benchmark Workflow
1. **File Ingestion:** Reading `wine_quality_merged.csv`.
2. **Filtering:** Filtering rows where `quality >= 6` and `alcohol > 10.0`.
3. **Aggregation:** Grouping by `quality` and computing the mean `alcohol` and mean `fixed acidity`.

### Benchmark Code Comparison

**Pandas Implementation:**
```python
df = pd.read_csv("Data/wine_quality_merged.csv")
filtered_df = df[(df["quality"] >= 6) & (df["alcohol"] > 10.0)]
agg_df = (
    filtered_df.groupby("quality")
    .agg({"alcohol": "mean", "fixed acidity": "mean"})
    .reset_index()
)
```

**Polars Implementation:**
```python
df = pl.read_csv("Data/wine_quality_merged.csv")
filtered_df = df.filter((pl.col("quality") >= 6) & (pl.col("alcohol") > 10.0))
agg_df = filtered_df.group_by("quality").agg(
    [
        pl.col("alcohol").mean().alias("alcohol_mean"),
        pl.col("fixed acidity").mean().alias("fixed_acidity_mean"),
    ]
)
```

### Benchmark Results

Timing was evaluated using high-resolution performance counters (`time.perf_counter()`):

| Framework | Execution Time | Speedup Factor |
|---|---|---|
| **Pandas** | **54.41 ms** | Baseline (1.0x) |
| **Polars** | **15.63 ms** | **~3.48x faster** |

### Why Polars Outperforms Pandas:
1. **Rust Core Engine:** Polars is implemented in Rust and built on the Apache Arrow column format, eliminating Python runtime overhead and optimizing CPU cache utilization.
2. **Multithreading:** While Pandas executes sequentially on a single thread by default, Polars automatically parallelizes operations across all available CPU cores.
3. **Expression Optimization:** The Polars expression API (`pl.col(...)`) avoids creating costly intermediate DataFrame copies during filtering and aggregation, directly computing final projections in memory.

---

## Deliverables & Repository Structure

- `analysis.py` — Standalone Python script containing the complete analytical pipeline, visualizations, and Polars vs. Pandas benchmark.
- `notebook.ipynb` — Original interactive exploration notebook.
- `Rust_testing.ipynb` — Jupyter notebook demonstrating Rust ownership, borrowing, and scope-based memory management.
- `Data/wine_quality_merged.csv` — Full merged dataset.
- `README.md` — Project documentation and analytical findings.


## Tools Used

- **Python 3.14** — core programming language for data analysis and ML baseline
- **Pandas** — data ingestion, inspection, filtering, and summary statistics
- **Polars** — high-performance, Rust-based DataFrame library used for execution speed benchmarking
- **scikit-learn** — data splitting (`train_test_split`), Linear Regression modeling, and metric evaluations (`mean_squared_error`, `r2_score`)
- **Matplotlib & Seaborn** — exploratory data visualization (distribution histograms and feature scatter plots)
- **Rust (via Evcxr Jupyter Kernel)** — experimentation with memory safety, move semantics, borrowing, and scoping rules
- **VS Code** — primary development environment (Terminal, Python Script, and Jupyter Notebook extension)