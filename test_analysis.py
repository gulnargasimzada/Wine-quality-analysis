import os
import pandas as pd
import pytest
from analysis import (
    evaluate_model,
    load_data,
    preprocess_data,
    run_pipeline,
    train_model,
)

CSV_FILE_PATH = "Data/wine_quality_merged.csv"


# ==========================================
# 1. UNIT TESTS (Vahid Testlər)
# ==========================================


def test_load_data():
    """Unit Test 1: Verify data loads correctly with expected columns."""
    assert os.path.exists(CSV_FILE_PATH), f"File not found: {CSV_FILE_PATH}"
    df = load_data(CSV_FILE_PATH)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "quality" in df.columns
    assert "alcohol" in df.columns


def test_preprocess_data_shapes():
    """Unit Test 2: Check train/test split proportions and target vector."""
    df = load_data(CSV_FILE_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # 80/20 train-test split verification
    total_samples = len(X_train) + len(X_test)
    assert len(X_test) / total_samples == pytest.approx(0.2, abs=0.01)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert "quality" not in X_train.columns


def test_preprocess_data_edge_case():
    """Unit Test 3 (Edge Case): Empty DataFrame should raise ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame cannot be empty."):
        preprocess_data(empty_df)


def test_train_and_evaluate_model():
    """Unit Test 4: Model evaluation metrics should stay within reasonable bounds."""
    df = load_data(CSV_FILE_PATH)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    mse, r2 = evaluate_model(model, X_test, y_test)

    # MSE must be non-negative, and R^2 score should be reasonable for this dataset
    assert mse > 0.0
    assert 0.0 <= r2 <= 1.0


# ==========================================
# 2. SYSTEM TEST (Bütöv Sistem Testi)
# ==========================================


def test_entire_pipeline_system():
    """System Test: Verify the entire end-to-end workflow completes successfully."""
    results = run_pipeline(CSV_FILE_PATH)

    assert "model" in results
    assert "mse" in results
    assert "r2" in results
    assert "speedup" in results
    assert results["speedup"] > 0.0