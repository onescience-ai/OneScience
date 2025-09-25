import importlib
import inspect
import logging
import json
from typing import List
from langchain_core.tools import BaseTool
from langchain_community.tools import ShellTool, ReadFileTool, WriteFileTool
from langchain_experimental.tools import PythonREPLTool
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from agent.llm import ChatModel
from agent.tools.description.tool_module_desc import description

PACKAGE_NAME = "agent.tool"
EXAMPLE_PACKAGE_NAME = "agent.tool.example"

logger = logging.getLogger(__name__)


def get_tools(tool_modules: list):
    if not tool_modules or not len(tool_modules):
        return [ShellTool(), PythonREPLTool(), ReadFileTool(), WriteFileTool()]
    else:
        application_tools = []
        application_tool_names = []
        application_tool_examples = []
        for module_name in tool_modules:
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

        for module_name in tool_modules:
            full_module_name = f"{EXAMPLE_PACKAGE_NAME}.{module_name}"
            module = importlib.import_module(full_module_name)
            application_tool_examples.extend(getattr(module, "examples", []))

        logger.info(f"application tools: {application_tool_names}")
        return application_tools, application_tool_examples


class ToolModulesResult(BaseModel):
    modules: List[str] = Field(description="匹配到的工具模块")


def match_tool_modules(prompt: str, chat_model_config: dict):
    system_prompt = """
你是一个工具模块选择助手。请根据用户任务的需求，从提供的可选工具模块中，精准匹配出完成该任务所必需的工具模块。

可使用工具模块详情：
{tool_module_info}

要求：
1. 仅选择与任务直接相关且必要的工具模块；
2. 输出必须为JSON格式，且仅包含一个字段 `modules`，其值为字符串数组（每个字符串为工具模块名）；
3. 不要包含任何解释、注释或额外文本；
4. 若无匹配模块，返回空数组。

请确保输出严格符合以下 JSON Schema：
{format_instructions}
"""
    llm = ChatModel[chat_model_config["factory_name"]](
        **chat_model_config["model"]
    )

    parser = JsonOutputParser(pydantic_object=ToolModulesResult)
    messages = [SystemMessage(system_prompt.format(
        tool_module_info="\n".join([f'工具模块名称：{k}，该模块的功能：{v}' for k, v in description.items()]),
        format_instructions=parser.get_format_instructions())), HumanMessage(prompt)]
    runnable = ChatPromptTemplate.from_messages(messages) | llm | parser

    result = runnable.invoke({})

    result = ToolModulesResult.model_validate_json(json.dumps(result))
    return result.modules


if __name__ == "__main__":
    get_tools({"tool_modules": ["test"]})
