from typing import List, Annotated, Dict, TypedDict

from ai_core.agent.unit.runner import Runner


class ReflectRunner(Runner):
    name = "reflection"
    system_prompt = """"
你需要对用户请求任务的执行过程及结果进行反思，要求：
- 请仔细审查之前的执行过程、推理和解决方案
- 严厉地批评哪些地方可以改进
- 请具体且有建设性
- 深入思考为完成该任务还缺少什么
- 不要提出问题，只提供反馈
"""

    def __init__(self, config: Dict):
        super().__init__(config)
