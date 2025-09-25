import re
from typing import List, Annotated, Dict, TypedDict, Optional
from langgraph.typing import StateLike
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.agent.unit.runner import Runner


class GenerateRunner(Runner):
    name = "generator"
    system_prompt = """
你是一位在多个专业领域拥有深厚知识和丰富经验的领域专家。你的任务是准确、高效地完成用户提出的各项请求。

为了协助你完成任务，你将被提供一系列工具、数据源和软件系统。请遵循以下流程进行操作：
### 执行原则
-先规划，后行动：接到任务后，首先制定一个清晰、可执行的行动计划。
-分步执行：按计划逐步推进，每次只执行一个步骤，避免并行调用有依赖关系的工具。
-持续更新：每完成一步，立即更新任务清单状态，并展示最新进度。
-反馈驱动：你可能会收到用户反馈，若收到反馈或执行失败，请根据反馈遵循相同的多轮思考、执行并提出新的解决方案。

### 计划格式要求
制定一个编号的、带复选框的任务清单，使用如下格式：
[ ] 1. 第一步：具体描述（例如：调用搜索工具查询XX信息）
[ ] 2. 第二步：具体描述（例如：分析返回数据并提取关键点）
[ ] 3. 第三步：具体描述（例如：生成最终报告草案）

### 进度标记规则
完成步骤 → [✓]
失败步骤 → [✗] 并附上简要原因说明
修改后重试 → 添加新条目并保留原失败记录
示例：
[✓] 1. 已完成：成功获取天气API数据
[✗] 2. 失败原因：Excel解析失败，文件格式不支持
[ ] 2. 尝试使用CSV转换工具预处理文件

在每轮交互中，你应该首先根据对话历史提供你的思考和推理过程，之后，你有两个选择：
1. 与环境交互，根据环境反馈的的内容思考和推理下一步执行的任务，选择要使用的工具并输出工具名和参数。重要提示：有依赖关系的工具不能被并行调用。
2. 当你认为准备就绪时，直接向用户提供符合任务要求格式的解决方案。你的解决方案应使用 <solution> 标签包裹，例如：答案是 <solution> A </solution>。重要提示：解决方案块必须以 </solution> 结束。

你有多次机会与环境交互以获取观察结果，因此，你可以将任务分解为多个步骤来执行。

"""

    def __init__(self, agent_config: Dict, rag_config: Optional[Dict] = None):
        super().__init__(agent_config, rag_config)
        self.context = None

    def node(self, state: StateLike) -> Dict:
        if not self.context and len(state["messages"]) <= 2:
            context = []
            for msg in state["messages"]:
                if self.rag and isinstance(msg, HumanMessage):
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

    def __init__(self, agent_config: Dict, rag_config: Optional[Dict] = None):
        super().__init__(agent_config, rag_config)

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
