"""
Test async fit_predict with custom data handles (Issue #7).

Verifies that fit_predict_async_tool supports both demo datasets
and custom data handles loaded via load_data_source.
"""

import sys
import asyncio
import unittest

sys.path.insert(0, "src")

from sktime_mcp.tools.fit_predict import fit_predict_async_tool
from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.jobs import get_job_manager, JobStatus


class TestAsyncWithDataset(unittest.TestCase):
    """Verify existing demo dataset path still works."""

    def test_async_with_dataset(self):
        """Demo dataset should return a job_id."""
        executor = get_executor()

        # instantiate a simple forecaster
        est = executor.instantiate("NaiveForecaster")
        self.assertTrue(est["success"])
        handle = est["handle"]

        result = fit_predict_async_tool(handle, dataset="airline", horizon=6)
        self.assertTrue(result["success"])
        self.assertIn("job_id", result)
        self.assertEqual(result["dataset"], "airline")


class TestAsyncWithDataHandle(unittest.TestCase):
    """Verify new custom data handle path works."""

    def test_async_with_data_handle(self):
        """Custom data handle should return a job_id."""
        executor = get_executor()

        # load a demo dataset through the normal code path so we
        # have a real data handle to work with
        data_result = executor.load_dataset("airline")
        self.assertTrue(data_result["success"])

        # manually store it as a data handle (same way load_data_source does)
        test_handle_id = "test_async_handle"
        executor._data_handles[test_handle_id] = {
            "y": data_result["data"],
            "X": data_result.get("exog"),
            "metadata": {"source": "test"},
            "validation": {"valid": True},
        }

        est = executor.instantiate("NaiveForecaster")
        handle = est["handle"]

        result = fit_predict_async_tool(handle, data_handle=test_handle_id, horizon=6)
        self.assertTrue(result["success"])
        self.assertIn("job_id", result)
        self.assertEqual(result["dataset"], test_handle_id)

        # cleanup
        del executor._data_handles[test_handle_id]


class TestAsyncValidation(unittest.TestCase):
    """Verify input validation for mutually exclusive params."""

    def test_both_provided_error(self):
        """Providing both dataset and data_handle should fail."""
        result = fit_predict_async_tool(
            "est_fake",
            dataset="airline",
            data_handle="data_xyz",
        )
        self.assertFalse(result["success"])
        self.assertIn("not both", result["error"])

    def test_neither_provided_error(self):
        """Providing neither dataset nor data_handle should fail."""
        result = fit_predict_async_tool("est_fake")
        self.assertFalse(result["success"])
        self.assertIn("required", result["error"])

    def test_invalid_data_handle(self):
        """Invalid data handle should be caught at the executor level."""
        executor = get_executor()
        job_manager = get_job_manager()

        est = executor.instantiate("NaiveForecaster")
        handle = est["handle"]

        # run the async method directly to test executor-level validation
        result = asyncio.run(
            executor.fit_predict_async(
                handle,
                dataset=None,
                horizon=6,
                data_handle="nonexistent_handle",
            )
        )
        self.assertFalse(result["success"])
        self.assertIn("Unknown data handle", result["error"])


if __name__ == "__main__":
    unittest.main()
