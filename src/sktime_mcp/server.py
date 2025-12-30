"""
MCP Server for sktime.

Main entry point for the Model Context Protocol server
that exposes sktime's registry and execution capabilities to LLMs.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from sktime_mcp.tools.list_estimators import (
    list_estimators_tool,
    get_available_tasks,
    get_available_tags,
)
from sktime_mcp.tools.describe_estimator import (
    describe_estimator_tool,
    search_estimators_tool,
)
from sktime_mcp.tools.instantiate import (
    instantiate_estimator_tool,
    instantiate_pipeline_tool,
    release_handle_tool,
    list_handles_tool,
)
from sktime_mcp.tools.fit_predict import (
    fit_predict_tool,
    fit_tool,
    predict_tool,
    list_datasets_tool,
)
from sktime_mcp.tools.codegen import export_code_tool
from sktime_mcp.composition.validator import get_composition_validator

# Configure logging to stderr with detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Create MCP server instance
server = Server("sktime-mcp")


def sanitize_for_json(obj):
    """Recursively convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    else:
        return obj



@server.list_tools()
async def list_tools() -> List[Tool]:
    """List all available MCP tools."""
    return [
        Tool(
            name="list_estimators",
            description="Discover sktime estimators by task type and capability tags",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Task type filter: forecasting, classification, regression, transformation, clustering",
                    },
                    "tags": {
                        "type": "object",
                        "description": "Filter by capability tags, e.g. {'capability:pred_int': true}",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 50)",
                        "default": 50,
                    },
                },
            },
        ),
        Tool(
            name="describe_estimator",
            description="Get detailed information about a specific sktime estimator",
            inputSchema={
                "type": "object",
                "properties": {
                    "estimator": {
                        "type": "string",
                        "description": "Name of the estimator (e.g., 'ARIMA', 'RandomForest')",
                    },
                },
                "required": ["estimator"],
            },
        ),
        Tool(
            name="instantiate_estimator",
            description="Create an estimator instance with given parameters",
            inputSchema={
                "type": "object",
                "properties": {
                    "estimator": {
                        "type": "string",
                        "description": "Name of the estimator to instantiate",
                    },
                    "params": {
                        "type": "object",
                        "description": "Hyperparameters for the estimator",
                    },
                },
                "required": ["estimator"],
            },
        ),
        Tool(
            name="instantiate_pipeline",
            description="Create a pipeline instance from a list of components",
            inputSchema={
                "type": "object",
                "properties": {
                    "components": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of estimator names in pipeline order (e.g., ['Detrender', 'ARIMA'])",
                    },
                    "params_list": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Optional list of hyperparameter dicts for each component",
                    },
                },
                "required": ["components"],
            },
        ),
        Tool(
            name="fit_predict",
            description="Fit an estimator on a dataset and generate predictions",
            inputSchema={
                "type": "object",
                "properties": {
                    "estimator_handle": {
                        "type": "string",
                        "description": "Handle from instantiate_estimator",
                    },
                    "dataset": {
                        "type": "string",
                        "description": "Dataset name: airline, sunspots, lynx, etc.",
                    },
                    "horizon": {
                        "type": "integer",
                        "description": "Forecast horizon (default: 12)",
                        "default": 12,
                    },
                },
                "required": ["estimator_handle", "dataset"],
            },
        ),
        Tool(
            name="validate_pipeline",
            description="Check if a pipeline composition is valid",
            inputSchema={
                "type": "object",
                "properties": {
                    "components": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of estimator names in pipeline order",
                    },
                },
                "required": ["components"],
            },
        ),
        Tool(
            name="list_datasets",
            description="List available demo datasets",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_available_tags",
            description="List all queryable capability tags",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="search_estimators",
            description="Search estimators by name or description",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="export_code",
            description="Export an estimator or pipeline as executable Python code",
            inputSchema={
                "type": "object",
                "properties": {
                    "handle": {
                        "type": "string",
                        "description": "Handle ID of the estimator/pipeline to export",
                    },
                    "var_name": {
                        "type": "string",
                        "description": "Variable name to use in generated code (default: 'model')",
                        "default": "model",
                    },
                    "include_fit_example": {
                        "type": "boolean",
                        "description": "Whether to include a fit/predict example (default: false)",
                        "default": False,
                    },
                },
                "required": ["handle"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    logger.info(f"=== Tool Call: {name} ===")
    logger.info(f"Arguments: {json.dumps(arguments, indent=2)}")
    
    try:
        if name == "list_estimators":
            result = list_estimators_tool(
                task=arguments.get("task"),
                tags=arguments.get("tags"),
                limit=arguments.get("limit", 50),
            )
        elif name == "describe_estimator":
            result = describe_estimator_tool(arguments["estimator"])
        elif name == "instantiate_estimator":
            result = instantiate_estimator_tool(
                arguments["estimator"],
                arguments.get("params"),
            )
        elif name == "instantiate_pipeline":
            result = instantiate_pipeline_tool(
                arguments["components"],
                arguments.get("params_list"),
            )
        elif name == "fit_predict":
            result = fit_predict_tool(
                arguments["estimator_handle"],
                arguments["dataset"],
                arguments.get("horizon", 12),
            )
            # Sanitize immediately to handle Period objects
            result = sanitize_for_json(result)
        elif name == "validate_pipeline":
            validator = get_composition_validator()
            validation = validator.validate_pipeline(arguments["components"])
            result = validation.to_dict()
        elif name == "list_datasets":
            result = list_datasets_tool()
        elif name == "get_available_tags":
            result = get_available_tags()
        elif name == "search_estimators":
            result = search_estimators_tool(
                arguments["query"],
                arguments.get("limit", 20),
            )
        elif name == "export_code":
            result = export_code_tool(
                arguments["handle"],
                arguments.get("var_name", "model"),
                arguments.get("include_fit_example", False),
            )
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        logger.info(f"=== Result for {name} ===")
        
        # Sanitize result for JSON serialization
        sanitized_result = sanitize_for_json(result)
        logger.info(f"{json.dumps(sanitized_result, indent=2, default=str)}")
        
        return [TextContent(type="text", text=json.dumps(sanitized_result, indent=2, default=str))]
    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def run_server():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Main entry point."""
    print("Starting sktime-mcp server...")
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
