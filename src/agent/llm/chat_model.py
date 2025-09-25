import os
import logging

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["QWEN_AGENT_MAX_LLM_CALL_PER_RUN"] = "1"

import re
import json
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from typing import Any, Optional, List, Union, Callable
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.language_models.base import (
    LanguageModelInput,
)
from langchain_core.tools import BaseTool
from langchain_core.messages.tool import ToolCall
from collections.abc import Sequence
from qwen_agent.agents import Assistant

logger = logging.getLogger(__name__)


class VLLMQwen3(Runnable):
    _FACTORY_NAME = "VLLMQwen3"

    def __init__(
        self,
        model_name: str,
        tools: Union[dict[str, Any], type, Callable, BaseTool] = None,
        **kwargs,
    ):

        from vllm import LLM
        from vllm.sampling_params import SamplingParams

        self.llm = LLM(model=model_name)
        self.tools = [convert_to_openai_tool(tool) for tool in tools or []]
        self.sampling_params = SamplingParams(
            max_tokens=kwargs["max_tokens"] if "max_tokens" in kwargs else 8192,
            temperature=kwargs["temperature"] if "temperature" in kwargs else 0.01,
        )

    def invoke(
        self,
        inputs: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        # input 是前一个组件的输出
        if isinstance(inputs, list) and isinstance(inputs[0], BaseMessage):
            prompt = inputs[-1].content
        else:
            prompt = inputs
        if isinstance(prompt, str):
            prompt = [prompt]
        if "schema" in kwargs:
            schema = kwargs["schema"]
            sampling_params = {
                **self.sampling_params,
                "guided_decoding_backend": "outlines",
                "guided_json": schema,
            }
        else:
            sampling_params = self.sampling_params

        outputs = self.llm.chat(prompt, sampling_params, tools=self.tools)
        ai_message = AIMessage(
            content=outputs.outputs[0].text,  # 提取生成的文本
            additional_kwargs={
                "generation_info": {
                    "finish_reason": outputs.outputs[0].finish_reason,
                    "logprobs": outputs.outputs[0].logprobs,
                }
            },
        )
        return ai_message


class HttpQwen3(Runnable):
    _FACTORY_NAME = "HttpQwen3"

    def __init__(
        self, model_name, tools: Union[dict[str, Any], type, Callable, BaseTool] = []
    ):
        self.model = model_name
        self.client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            # api_key="sk-6cde1152770148dda5ecff3d69bb8631",
            api_key="sk-cf71112ee4074968bb6af6e27fb3d2f0",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.tools = [convert_to_openai_tool(tool) for tool in tools]
        self.mapping = {
            "system": "system",
            "human": "user",
            "assistant": "assistant",
            "ai": "assistant",
            "tool": "tool",
            "function": "function",
        }

    def invoke(
        self,
        inputs: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(inputs, PromptValue):
            messages = []
            for inp in inputs.messages:
                if isinstance(inp, ToolMessage):
                    messages.append(
                        {
                            "role": "tool",
                            "content": inp.content,
                            "tool_call_id": inp.tool_call_id,
                        }
                    )
                elif isinstance(inp, AIMessage) and len(inp.tool_calls):
                    tool_calls = [
                        {
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call["args"]),
                            },
                        }
                        for tool_call in inp.tool_calls
                    ]
                    messages.append(
                        {
                            "role": "assistant",
                            "content": inp.content,
                            "tool_calls": tool_calls,
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": self.mapping[inp.model_fields["type"].default],
                            "content": inp.content,
                        }
                    )
        elif isinstance(inputs, list) and isinstance(inputs[0], BaseMessage):
            messages = []
            for inp in inputs:
                if isinstance(inp, ToolMessage):
                    messages.append(
                        {
                            "role": "tool",
                            "content": inp.content,
                            "tool_call_id": inp.tool_call_id,
                        }
                    )
                elif isinstance(inp, AIMessage) and len(inp.tool_calls):
                    tool_calls = [
                        {
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call["args"]),
                            },
                        }
                        for tool_call in inp.tool_calls
                    ]
                    messages.append(
                        {
                            "role": "assistant",
                            "content": inp.content,
                            "tool_calls": tool_calls,
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": self.mapping[inp.model_fields["type"].default],
                            "content": inp.content,
                        }
                    )
        elif isinstance(inputs, str):
            messages = [{"role": "system", "content": inputs}]
        else:
            messages = inputs

        if len(self.tools):
            completion = self.client.chat.completions.create(
                # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                model=self.model,
                messages=messages,
                # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
                # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
                tools=self.tools,
                extra_body={"enable_thinking": False},
            )
        else:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body={"enable_thinking": False},
            )

        ai_message = AIMessage(content=completion.choices[0].message.content)
        if completion.choices[0].finish_reason == "tool_calls":
            ai_message.tool_calls = [
                ToolCall(
                    id=tool_call.id,
                    args=json.loads(tool_call.function.arguments),
                    name=tool_call.function.name,
                    type="tool_call",
                )
                for tool_call in completion.choices[0].message.tool_calls
            ]
        return ai_message

class LocalQwen3(Runnable):
    _FACTORY_NAME = "LocalQwen3"

    def __init__(self, model_name: str, model_server: str, tools: list):
        llm_cfg = {
            # Use a model service compatible with the OpenAI API, such as vLLM or Ollama:
            "model": model_name,
            "model_server": model_server,  # base_url, also known as api_base
            "api_key": "EMPTY",
            # (Optional) LLM hyperparameters for generation:
            "generate_cfg": {
                "top_p": 0.8,
                "max_input_tokens": 20480,
                "temperature": 0.01,
            },
        }
        tools = [convert_to_openai_tool(tool)["function"] for tool in tools]
        print(llm_cfg)
        self.agent = Assistant(
            llm=llm_cfg,
            system_message="",
            # function_list=tools
        )
        self.mapping = {
            "system": "system",
            "human": "user",
            "assistant": "assistant",
            "ai": "assistant",
        }
        self.function_call = """
可使用的工具
{tools}

工具调用输出格式如下：

```json
{{
  "type": "function",
  "name": "write_file",
  "parameters": {{
    "file_path": "spring.txt",
    "text": "春日暖阳，万物复苏。嫩绿的草芽破土而出，桃花绽开笑颜，燕子呢喃穿梭。微风轻拂，带来泥土的芬芳与生命的气息，大地披上绚丽的外衣，生机勃勃，令人心旷神怡。",
    "append": false
  }}
}}
```
        """.format(
            tools=tools
        )

    def invoke(
        self,
        inputs: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(inputs, PromptValue):
            messages = []
            for inp in inputs.messages:
                if isinstance(inp, ToolMessage):
                    messages.append(
                        {
                            "role": "tool",
                            "content": inp.content,
                            "tool_call_id": inp.tool_call_id,
                        }
                    )
                elif isinstance(inp, AIMessage) and len(inp.tool_calls):
                    tool_calls = [
                        {
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call["args"]),
                            },
                        }
                        for tool_call in inp.tool_calls
                    ]
                    messages.append(
                        {
                            "role": "assistant",
                            "content": inp.content,
                            "tool_calls": tool_calls,
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": self.mapping[inp.model_fields["type"].default],
                            "content": inp.content,
                        }
                    )
        elif isinstance(inputs, list) and isinstance(inputs[0], BaseMessage):
            messages = []
            for inp in inputs:
                if isinstance(inp, ToolMessage):
                    messages.append(
                        {
                            "role": "function",
                            "content": inp.content,
                            "tool_call_id": inp.tool_call_id,
                        }
                    )
                elif isinstance(inp, AIMessage) and len(inp.tool_calls):
                    tool_calls = [
                        {
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call["args"]),
                            },
                        }
                        for tool_call in inp.tool_calls
                    ]
                    messages.append(
                        {
                            "role": "assistant",
                            "content": inp.content,
                            "tool_calls": tool_calls,
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": self.mapping[inp.model_fields["type"].default],
                            "content": inp.content,
                        }
                    )
        elif isinstance(inputs, str):
            messages = [{"role": "system", "content": inputs}]
        else:
            messages = inputs

        if messages[0]["role"] == "system":
            messages[0]["content"] = messages[0]["content"] + self.function_call
        logger.info(f"Input messages: {messages}")

        for responses in self.agent.run(messages=messages):
            pass
        logger.info(f"LLM response: {responses}")

        content = ""
        tool_calls = []
        for res in responses:
            if res["role"] != "assistant":
                continue
            # think_match = re.search(r"<think>(.*?)</think>", res["content"], re.DOTALL)
            # content = f"{content}\n{think_match.group(1) if think_match else ''}"
            content = f"{content}\n{res['content']}"
            functions = re.findall(
                r"```\s*json\s*(.*?)\s*```", res["content"], re.DOTALL
            )
            print(f"functions ****{functions}")
            functions = [json.loads(function) for function in functions]
            functions = [
                function for function in functions if function["type"] == "function"
            ]
            tool_calls.extend(
                [
                    ToolCall(
                        id=str(hash(func["name"])),
                        name=func["name"],
                        args=func.get("parameters", {}),
                    )
                    for func in functions
                ]
            )
        msg = AIMessage(content=content.strip(), tool_calls=tool_calls)
        return msg


class HuggingfaceLLM:
    _FACTORY_NAME = "HuggingfaceLLM"

    def __init__(
        self,
        model_name: str,
        tools: Sequence[Union[dict[str, Any], type, Callable, BaseTool]] = None,
        **kwargs: Any,
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name,
            trust_remote_code=True,
            cache_dir=kwargs.get("cache_dir", "../../agent/download"),
        )
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=kwargs.get("cache_dir", "../../agent/download"),
        )
        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=kwargs.get(
                "max_new_tokens", kwargs.get("max_new_tokens", 1024)
            ),
        )
        hugging_pipe = HuggingFacePipeline(pipeline=pipe)
        self.llm = ChatHuggingFace(llm=hugging_pipe).bind_tools(tools or [])

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> BaseMessage:
        result = self.llm.invoke(input=input, config=config, stop=stop, **kwargs)
        content = result.content
        if not content.endswith("<|im_end|>"):
            content = content + "<|im_end|>"
        msgs = re.findall(r"<\|im_start\|>(.*?)<\|im_end\|>", content, re.DOTALL)
        if msgs:
            msg = msgs[-1]
            if msg.startswith("assistant"):
                msg = msg[len("assistant") :].strip()
            result.content = msg
        return result


if __name__ == "__main__":
    from agent.llm import ChatModel
    from agent.agent.unit.tool_manager import get_tools

    tools = get_tools()
    llm = ChatModel["HuggingfaceLLM"](
        model_name="../download/models--Qwen--Qwen3-4B/snapshots", tools=tools
    )
    messages = [
        {
            "role": "system",
            "content": "\n你是一个任务执行专家,请根据用户给定的总任务和待执行的子任务执行当前子任务，要求：\n1. 理解和分析要执行的子任务，识别它是否需要调用工具来执行\n2. 基于1分析到当前任务需要调用工具来执行，则根据任务要求和之前执行的子任务的执行结果生成调用工具信息\n3. 基于1分析到当前任务不需要调用工具来执行或者输入已经工具执行结果，则直接根据输入和任务要求生成结果\n\n例:\n任务：将生成的文本'春，是万物复苏的季节。微风拂面，阳光温暖，花草树木竞相绽放生机。田野间绿意盎然，鸟儿欢唱，仿佛整个世界都在迎接新生。春天不仅带来了自然的美景，也唤起了人们对未来的希望与憧憬'写入'D:/file/spring.txt'文件中\n任务执行：分析到当前任务是将文本写入文件中，这个需要调用文本写出的工具来执行，又识别到write_file工具可实现该功能，所以调用write_file工具来执行任务，参数说明，file_path:'D:/file/spring.txt\",'text':'春，是万物复苏的季节。微风拂面，阳光温暖，花草树木竞相绽放生机。田野间绿意盎然，鸟儿欢唱，仿佛整个世界都在迎接新生。春天不仅带来了自然的美景，也唤起了人们对未来的希望与憧憬'\n\n",
        },
        {
            "role": "user",
            "content": "总任务：\n请生成主题为春，总字数不超100字,并写入D:/molsculptor_agent/agent/config/test.txt文件中\n当前任务：\n请生成主题为春的字数不超过100字的一段话\n之前的子任务执行结果：\n[{'plan_task': '生成围绕春这个主题的一段话，总字数不超过100字', 'execute_result': '春，是万物复苏的季节。微风拂面，阳光温暖，花草树木竞相绽放生机。田野间绿意盎然，鸟儿欢唱，仿佛整个世界都在迎接新生。春天不仅带来了自然的美景，也唤起了人们对未来的希望与憧憬。'}]\n之前执行错误子任务原因：\n[]\n",
        },
    ]
    result = llm.invoke(messages)
    print(result)
    print("----------------------------")
