import re
from typing import List, Annotated, Dict, TypedDict
from langgraph.typing import StateLike
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_core.agent.unit.runner import Runner


class GenerateRunner(Runner):
    name = "generator"
    system_prompt = """
你是一位领域专家，在多个专业领域都有深厚的知识和丰富的经验，你需要完成用户输入的任务

为此，会给你配备各种工具函数、数据和软件，以协助你完成整个过程。

在接到一项任务后，首先制定一个计划。该计划应是一个带编号的步骤清单，详细说明你将如何解决该任务。请确保内容具体且详尽。
将你的计划格式化为一个带复选框的清单，如下所示：

 第一步
 第二步
 第三步
然后按照计划逐步执行。完成每个步骤后，通过将空复选框替换为勾选标记来更新清单：

[✓] 第一步（已完成）
 第二步
 第三步
如果某个步骤失败或需要修改，请将其标记为 X 并解释原因：

[✓] 第一步（已完成）
[✗] 第二步（失败原因：...）
 修改后的第二步
 第三步
每次执行后都应展示更新后的计划，以便用户跟踪进度。

在每一轮交互中，你应该首先根据对话历史提供你的思考和推理过程，之后，你有两个选择：

1. 与环境交互，根据环境反馈的的内容思考和推理下一步执行的任务，选择要使用的工具并输出工具名和参数。重要提示：有依赖关系的工具不能被并行调用。
2. 当你认为准备就绪时，直接向用户提供符合任务要求格式的解决方案。你的解决方案应使用 <solution> 标签包裹，例如：答案是 <solution> A </solution>。重要提示：解决方案块必须以 </solution> 结束。

你有多次机会与环境交互以获取观察结果，因此，你可以将任务分解为多个步骤来执行。

你可能会收到也可能不会收到人类的反馈，如果收到反馈，请根据反馈遵循相同的多轮思考、执行并提出新解决方案的流程。

"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.context = None

    def node(self, state: StateLike) -> Dict:
        if not self.context and len(state["messages"]) <= 2:
            context = ""
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    context = self.rag.retrieve(
                        user=self.user,
                        query=self.context,
                        retrieval_method="hybrid_search",
                    )
            context = "".join(context)
            example_texts = "\n".join(self.tool_examples).strip()
            if example_texts != "":
                context = f"{context}\n\n工具使用示例\n{example_texts}".strip()
            if len(context):
                self.context = context
        if self.context:
            messages = [
                SystemMessage(self.system_prompt + "\n\n" + self.context)
            ] + state["messages"]
        else:
            messages = [SystemMessage(self.system_prompt)] + state["messages"]
        response = self.llm.invoke(messages)
        self.logger.info(f"{self.name} response: {response}")
        return {"messages": [response]}


class ParseRunner(Runner):

    name = "parser"

    def __init__(self, config: Dict):
        super().__init__(config)

    def node(self, state: StateLike) -> Dict:
        msg = state["messages"][-1]
        tool_calls = msg.tool_calls if isinstance(msg, AIMessage) else []

        msg = msg.content
        if "<solution>" in msg and "</solution>" not in msg:
            msg += "</solution>"
        if "<think>" in msg and "</think>" not in msg:
            msg += "</think>"

        think_match = re.search(r"<think>(.*?)</think>", msg, re.DOTALL)
        answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL)

        if answer_match or think_match or len(tool_calls):
            return {"messages": []}
        else:
            # Check if we already added an error message to avoid infinite loops
            error_count = sum(
                1
                for m in state["messages"]
                if isinstance(m, AIMessage) and "没有这些标签" in m.content
            )

            if error_count >= 2:
                # Add a final message explaining the termination
                return {
                    "messages": AIMessage(
                        content="由于重复出现解析错误，执行已终止。请检查您的输入并重试。"
                    )
                }

            else:
                # Try to correct it
                return {
                    "messages": HumanMessage(
                        content="每个回复必须包含思考过程，后跟 <think> 或 <solution> 标签。但当前回复中没有这些标签。请遵循指令，修正并重新生成回复。"
                    )
                }
