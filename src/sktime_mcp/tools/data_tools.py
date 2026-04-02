"""
Data loading tools for sktime MCP.

Provides tools for loading data from various sources.
"""

from typing import Any, Dict
from sktime_mcp.runtime.executor import get_executor


def load_data_source_tool(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load data from any source (pandas, SQL, file, etc.).
    
    Args:
        config: Data source configuration
            {
                "type": "pandas" | "sql" | "file",
                ... (type-specific configuration)
            }
    
    Returns:
        Dictionary with:
        - success: bool
        - data_handle: str (handle ID for the loaded data)
        - metadata: dict (information about the data)
        - validation: dict (validation results)
    
    Examples:
        # Pandas DataFrame
        >>> load_data_source_tool({
        ...     "type": "pandas",
        ...     "data": {"date": [...], "value": [...]},
        ...     "time_column": "date",
        ...     "target_column": "value"
        ... })
        
        # SQL Database
        >>> load_data_source_tool({
        ...     "type": "sql",
        ...     "connection_string": "postgresql://user:pass@host:5432/db",
        ...     "query": "SELECT date, value FROM sales",
        ...     "time_column": "date",
        ...     "target_column": "value"
        ... })
        
        # CSV File
        >>> load_data_source_tool({
        ...     "type": "file",
        ...     "path": "/path/to/data.csv",
        ...     "time_column": "date",
        ...     "target_column": "value"
        ... })
    """
    executor = get_executor()
    return executor.load_data_source(config)


async def load_data_source_async_tool(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load data from any source in the background (non-blocking).
    
    This tool schedules the data loading as a background job and returns immediately
    with a job_id. Use check_job_status to monitor progress.
    
    Args:
        config: Data source configuration (same as load_data_source)
    
    Returns:
        Dictionary with:
        - success: bool
        - job_id: Job ID for tracking progress
        - message: Information about the job
    
    Example:
        >>> load_data_source_async_tool({
        ...     "type": "file",
        ...     "path": "/path/to/large_data.csv",
        ...     "time_column": "date",
        ...     "target_column": "value"
        ... })
        {
            "success": True,
            "job_id": "abc-123-def-456",
            "message": "Data loading started. Use check_job_status to monitor."
        }
    """
    import asyncio
    from sktime_mcp.runtime.jobs import get_job_manager
    
    executor = get_executor()
    job_manager = get_job_manager()
    
    source_type = config.get("type", "unknown")
    
    job_id = job_manager.create_job(
        job_type="data_loading",
        total_steps=3,
    )
    
    asyncio.create_task(executor.load_data_source_async(config, job_id))
    
    return {
        "success": True,
        "job_id": job_id,
        "message": f"Data loading started for '{source_type}'. Use check_job_status('{job_id}') to monitor.",
        "source_type": source_type,
    }


def list_data_sources_tool() -> Dict[str, Any]:
    """
    List all available data source types.
    
    Returns:
        Dictionary with:
        - success: bool
        - sources: list of available source types
        - descriptions: dict with descriptions for each source type
    """
    from sktime_mcp.data import DataSourceRegistry
    
    sources = DataSourceRegistry.list_adapters()
    
    # Get descriptions for each source
    descriptions = {}
    for source_type in sources:
        info = DataSourceRegistry.get_adapter_info(source_type)
        descriptions[source_type] = {
            "class": info["class"],
            "description": info["docstring"].split("\n")[0] if info["docstring"] else "",
        }
    
    return {
        "success": True,
        "sources": sources,
        "descriptions": descriptions,
    }


def fit_predict_with_data_tool(
    estimator_handle: str,
    data_handle: str,
    horizon: int = 12,
) -> Dict[str, Any]:
    """
    Fit and predict using custom data.
    
    Args:
        estimator_handle: Handle from instantiate_estimator
        data_handle: Handle from load_data_source
        horizon: Forecast horizon (default: 12)
    
    Returns:
        Dictionary with predictions
    
    Example:
        >>> fit_predict_with_data_tool(
        ...     estimator_handle="est_abc123",
        ...     data_handle="data_xyz789",
        ...     horizon=12
        ... )
    """
    executor = get_executor()
    return executor.fit_predict_with_data(
        estimator_handle,
        data_handle,
        horizon,
    )


def list_data_handles_tool() -> Dict[str, Any]:
    """
    List all loaded data handles.
    
    Returns:
        Dictionary with:
        - success: bool
        - count: int (number of loaded data handles)
        - handles: list of data handle information
    """
    executor = get_executor()
    return executor.list_data_handles()


def release_data_handle_tool(data_handle: str) -> Dict[str, Any]:
    """
    Release a data handle and free memory.
    
    Args:
        data_handle: Data handle to release
    
    Returns:
        Dictionary with success status
    """
    executor = get_executor()
    return executor.release_data_handle(data_handle)
