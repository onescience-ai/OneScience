import json


def parse_react_gent_log(save_path: str, log_path="../logs/react_agent.log"):
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    wf = open(save_path, "w", encoding="utf-8")
    before_input_messages = []
    for line in lines:
        if "ai_core.llm.chat_model - INFO - Input messages: " in line:
            input_messages = line.split(
                "ai_core.llm.chat_model - INFO - Input messages: "
            )[1].strip()
            input_messages = eval(input_messages)
            wf.write("【Input messages】\n")
            wf.write(
                json.dumps(
                    input_messages[len(before_input_messages) :],
                    indent=4,
                    ensure_ascii=False,
                )
            )
            before_input_messages = input_messages
            wf.write("\n\n")
        elif "ai_core.llm.chat_model - INFO - LLM response: " in line:
            responses = line.split("ai_core.llm.chat_model - INFO - LLM response: ")[
                1
            ].strip()
            responses = eval(responses)
            wf.write("【LLM responses】\n")
            wf.write(json.dumps(responses, indent=4, ensure_ascii=False))
            wf.write("\n\n")
        elif "ai_core.agent.unit.runner - INFO - generator response: " in line:
            response = line.split(
                "ai_core.agent.unit.runner - INFO - generator response: "
            )[1].strip()
            wf.write("【generator response】\n")
            wf.write(response)
            wf.write("\n\n")
        elif "ai_core.agent.unit.runner - INFO - tool_runner: " in line:
            tool_runner = line.split(
                "ai_core.agent.unit.runner - INFO - tool_runner: "
            )[1].strip()
            wf.write("【tool_runner】\n")
            wf.write(tool_runner)
            wf.write("\n\n")

    wf.close()


parse_react_gent_log(save_path="../logs/molsculptor_exp1.txt")
