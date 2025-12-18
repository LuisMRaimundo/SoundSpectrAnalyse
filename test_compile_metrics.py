import pytest
import pandas as pd
from compile_metrics import read_excel_metrics, apply_weighted_combination

def test_read_excel_metrics_with_weight_function(tmp_path):
    """
    Tests that read_excel_metrics can handle a 'Weight Function' column
    without raising a ValueError.
    """
    # Create a dummy Excel file
    excel_path = tmp_path / "test.xlsx"
    df = pd.DataFrame({
        "Note": ["A4"],
        "Density Metric": [0.5],
        "Weight Function": ["log"]
    })
    writer = pd.ExcelWriter(excel_path)
    df.to_excel(writer, sheet_name="Metrics", index=False)
    writer.close()

    # Call the function and assert that no exception is raised
    try:
        metrics = read_excel_metrics(excel_path)
        # Weight Function is in TEXT_FIELDS in compile_metrics.py, so it should be preserved
        assert isinstance(metrics, dict)
    except ValueError:
        pytest.fail("read_excel_metrics raised a ValueError unexpectedly.")

def test_apply_weighted_combination_removes_weight_function_col():
    """
    Tests that apply_weighted_combination removes the 'Weight Function'
    column before processing.
    """
    df = pd.DataFrame({
        "Spectral Density Metric": [0.5],
        "Filtered Density Metric": [0.5],
        "Weight Function": ["log"]
    })

    result_df = apply_weighted_combination(df, weight_function="log")
    assert "WF_used" in result_df.columns
    assert result_df["WF_used"].iloc[0].strip().lower() in {"log", "linear"}
