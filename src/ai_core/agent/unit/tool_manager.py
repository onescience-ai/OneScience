import importlib
import inspect
import logging
from langchain_core.tools import BaseTool
from langchain_community.tools import ShellTool, ReadFileTool, WriteFileTool
from langchain_experimental.tools import PythonREPLTool

PACKAGE_NAME = "ai_core.tool"
EXAMPLE_PACKAGE_NAME = "ai_core.tool.example"

logger = logging.getLogger(__name__)


def get_tools(config: dict):
    if "tool_modules" not in config:
        return [ShellTool(), PythonREPLTool(), ReadFileTool(), WriteFileTool()]
    else:
        application_tools = []
        application_tool_names = []
        application_tool_examples = []
        for module_name in config["tool_modules"]:
            full_module_name = f"{PACKAGE_NAME}.{module_name}"
            module = importlib.import_module(full_module_name)

            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and inspect.getmro(obj)[1] == BaseTool:
                    ins = obj()
                    application_tools.append(ins)
                    application_tool_names.append(ins.name)
                elif (
                    not inspect.isfunction(obj)
                    and not inspect.isclass(obj)
                    and isinstance(obj, BaseTool)
                ):
                    application_tools.append(obj)
                    application_tool_names.append(obj.name)

        for module_name in config["tool_modules"]:
            full_module_name = f"{EXAMPLE_PACKAGE_NAME}.{module_name}"
            module = importlib.import_module(full_module_name)
            application_tool_examples.extend(getattr(module, "examples", []))

        logger.info(f"application tools: {application_tool_names}")
        return application_tools, application_tool_examples


if __name__ == "__main__":
    get_tools({"tool_modules": ["test"]})
