"""
Tests for the unified fit_predict tool.

Verifies that fit_predict_tool handles both demo datasets
and custom data handles through a single interface.
"""

import sys

sys.path.insert(0, "src")

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.tools.data_tools import fit_predict_with_data_tool
from sktime_mcp.tools.fit_predict import fit_predict_tool


class TestUnifiedFitPredict:
    """Tests for the unified fit_predict_tool."""

    def test_fit_predict_with_dataset(self):
        """Using a demo dataset should work as before."""
        executor = get_executor()
        est = executor.instantiate("NaiveForecaster", {"strategy": "last"})
        assert est["success"]

        result = fit_predict_tool(est["handle"], dataset="airline")
        assert result["success"]
        assert "predictions" in result

    def test_fit_predict_with_data_handle(self):
        """Using a data_handle from load_data_source should work."""
        import pandas as pd

        executor = get_executor()
        est = executor.instantiate("NaiveForecaster", {"strategy": "last"})
        assert est["success"]

        config = {
            "type": "pandas",
            "data": {
                "date": pd.date_range("2020-01-01", periods=30, freq="D"),
                "value": list(range(30)),
            },
            "time_column": "date",
            "target_column": "value",
        }
        data_result = executor.load_data_source(config)
        assert data_result["success"]

        handle = data_result.get("data_handle")
        result = fit_predict_tool(est["handle"], data_handle=handle)
        assert result["success"]
        assert "predictions" in result

    def test_fit_predict_both_provided_error(self):
        """Providing both dataset and data_handle should fail."""
        result = fit_predict_tool(
            "est_fake",
            dataset="airline",
            data_handle="data_fake",
        )
        assert not result["success"]
        assert "not both" in result["error"]

    def test_fit_predict_neither_provided_error(self):
        """Providing neither dataset nor data_handle should fail."""
        result = fit_predict_tool("est_fake")
        assert not result["success"]
        assert "required" in result["error"]

    def test_fit_predict_invalid_data_handle(self):
        """An invalid data_handle should return a clear error."""
        executor = get_executor()
        est = executor.instantiate("NaiveForecaster", {"strategy": "last"})
        assert est["success"]

        result = fit_predict_tool(est["handle"], data_handle="bogus_handle")
        assert not result["success"]
        assert "bogus_handle" in result["error"]


class TestDeprecatedToolStillWorks:
    """The old fit_predict_with_data_tool should still function."""

    def test_deprecated_tool_still_works(self):
        """Backward compatibility: old tool still works."""
        import pandas as pd

        executor = get_executor()
        est = executor.instantiate("NaiveForecaster", {"strategy": "last"})
        assert est["success"]

        config = {
            "type": "pandas",
            "data": {
                "date": pd.date_range("2020-01-01", periods=30, freq="D"),
                "value": list(range(30)),
            },
            "time_column": "date",
            "target_column": "value",
        }
        data_result = executor.load_data_source(config)
        assert data_result["success"]

        handle = data_result.get("data_handle")
        result = fit_predict_with_data_tool(est["handle"], handle)
        assert result["success"]
        assert "predictions" in result
