from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ...exceptions import AgentsException, UserError
from ...logger import logger
from ...run_context import RunContextWrapper  # For empty context
from ...tool import (
    FunctionTool,
    Tool,
)  # Assuming Tool is the base, FunctionTool has on_invoke_tool
from .model import RealtimeEventToolCall  # The event type that triggers this


class ToolExecutor:
    """Executes tools based on RealtimeEventToolCall events."""

    def __init__(self, tools: Sequence[Tool]):
        self._tool_map: dict[str, FunctionTool] = {}
        for tool in tools:
            if isinstance(tool, FunctionTool):
                self._tool_map[tool.name] = tool
            else:
                # For now, only FunctionTools are supported by this simple executor.
                # We can extend this later if other tool types (e.g. ComputerTool) are needed
                # in the realtime flow directly without going through a full agent run.
                logger.warning(
                    f"Tool '{tool.name}' is not a FunctionTool and will be ignored by ToolExecutor."
                )

    async def execute(self, tool_call_event: RealtimeEventToolCall) -> str:
        """Executes the specified tool and returns its string output.

        Args:
            tool_call_event: The RealtimeEventToolCall describing the tool to execute.

        Returns:
            A string representation of the tool's output (typically JSON).

        Raises:
            AgentsException: If the tool is not found or fails during execution.
        """
        tool = self._tool_map.get(tool_call_event.tool_name)
        if not tool:
            err_msg = f"Tool '{tool_call_event.tool_name}' not found in ToolExecutor."
            logger.error(err_msg)
            # Return an error string that can be sent back to the LLM
            return json.dumps(
                {"error": err_msg, "tool_name": tool_call_event.tool_name}
            )

        # Convert arguments dict to JSON string, as expected by on_invoke_tool
        try:
            arguments_json = json.dumps(tool_call_event.arguments)
        except TypeError as e:  # pragma: no cover
            err_msg = f"Failed to serialize arguments for tool '{tool.name}': {e}"
            logger.error(f"{err_msg} Arguments: {tool_call_event.arguments}")
            return json.dumps({"error": err_msg, "tool_name": tool.name})

        logger.info(f"Executing tool: {tool.name} with args: {arguments_json}")

        try:
            # Create an empty RunContextWrapper for now, as this executor is lightweight.
            # If context-dependent tools are needed, this might need to evolve or use a proper Runner.
            # The `on_invoke_tool` is expected to handle JSON string input.
            tool_output = await tool.on_invoke_tool(
                RunContextWrapper(context=None), arguments_json
            )

            # Ensure the output is a string (as expected by OpenAI tool result content)
            if not isinstance(tool_output, str):
                # Attempt to convert common types to string (e.g. dict to JSON string)
                if isinstance(tool_output, (dict, list)):
                    tool_output_str = json.dumps(tool_output)
                else:
                    tool_output_str = str(tool_output)
            else:
                tool_output_str = tool_output

            logger.info(
                f"Tool {tool.name} executed successfully. Output length: {len(tool_output_str)}"
            )
            return tool_output_str
        except Exception as e:  # pragma: no cover
            logger.error(f"Error executing tool '{tool.name}': {e}", exc_info=True)
            # Return an error string that can be sent back to the LLM
            return json.dumps({"error": str(e), "tool_name": tool.name})
