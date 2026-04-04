"""
Tests for the unified fit_predict tool.
"""

import sys

import pytest

sys.path.insert(0, "src")


class TestUnifiedFitPredict:
    """Tests for fit_predict_tool accepting both dataset and data_handle."""

    def _get_estimator_handle(self):
        """Create a NaiveForecaster handle for reuse."""
        from sktime_mcp.runtime.executor import get_executor

        executor = get_executor()
        result = executor.instantiate("NaiveForecaster", {"strategy": "last"})
        assert result["success"], f"Failed to instantiate: {result}"
        return result["handle"]

    def test_fit_predict_with_dataset(self):
        """Passing a demo dataset name should work."""
        from sktime_mcp.tools.fit_predict import fit_predict_tool

        handle = self._get_estimator_handle()
        result = fit_predict_tool(
            estimator_handle=handle,
            dataset="airline",
            horizon=3,
        )

        assert result["success"], f"Expected success, got: {result}"
        assert "predictions" in result
        assert result["horizon"] == 3

    def test_fit_predict_with_data_handle(self):
        """Passing a data_handle from load_data_source should work."""
        import pandas as pd

        from sktime_mcp.runtime.executor import get_executor
        from sktime_mcp.tools.fit_predict import fit_predict_tool

        executor = get_executor()

        config = {
            "type": "pandas",
            "data": {
                "date": pd.date_range("2020-01-01", periods=50, freq="D").tolist(),
                "sales": [100 + i for i in range(50)],
            },
            "time_column": "date",
            "target_column": "sales",
        }
        load_result = executor.load_data_source(config)
        assert load_result["success"], f"Data load failed: {load_result}"

        handle = self._get_estimator_handle()
        result = fit_predict_tool(
            estimator_handle=handle,
            data_handle=load_result["data_handle"],
            horizon=5,
        )

        assert result["success"], f"Expected success, got: {result}"
        assert "predictions" in result

    def test_fit_predict_both_provided_error(self):
        """Providing both dataset and data_handle should fail."""
        from sktime_mcp.tools.fit_predict import fit_predict_tool

        handle = self._get_estimator_handle()
        result = fit_predict_tool(
            estimator_handle=handle,
            dataset="airline",
            data_handle="data_fake123",
            horizon=3,
        )

        assert not result["success"]
        assert "error" in result
        assert "not both" in result["error"].lower()

    def test_fit_predict_neither_provided_error(self):
        """Omitting both dataset and data_handle should fail."""
        from sktime_mcp.tools.fit_predict import fit_predict_tool

        handle = self._get_estimator_handle()
        result = fit_predict_tool(
            estimator_handle=handle,
            horizon=3,
        )

        assert not result["success"]
        assert "error" in result

    def test_deprecated_tool_still_works(self):
        """The old fit_predict_with_data_tool should still function."""
        import pandas as pd

        from sktime_mcp.runtime.executor import get_executor
        from sktime_mcp.tools.data_tools import fit_predict_with_data_tool

        executor = get_executor()

        config = {
            "type": "pandas",
            "data": {
                "date": pd.date_range("2020-01-01", periods=50, freq="D").tolist(),
                "sales": [100 + i for i in range(50)],
            },
            "time_column": "date",
            "target_column": "sales",
        }
        load_result = executor.load_data_source(config)
        assert load_result["success"]

        handle = self._get_estimator_handle()
        result = fit_predict_with_data_tool(
            estimator_handle=handle,
            data_handle=load_result["data_handle"],
            horizon=5,
        )

        assert result["success"], f"Deprecated tool failed: {result}"
        assert "predictions" in result
        assert "deprecation_notice" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
