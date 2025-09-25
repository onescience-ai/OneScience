import pprint
import urllib.parse
import json5
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.output_beautify import typewriter_print
import os

# Step 2: Configure the LLM you are using.
llm_cfg = {
    'model': 'qwen3_32B',
    'model_server': 'http://localhost:8000/v1',  # base_url, also known as api_base
    'api_key': 'EMPTY',

    'generate_cfg': {
        'top_p': 0.8,
        'thought_in_content': False
    },
}

def get_all_file(path):
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
    return all_files

# Step 3: Create an agent. Here we use the `Assistant` agent as an example, which is capable of using tools and reading files.
system_instruction = '''你是一个聊天机器人，请你根据文档的内容回答问题
- 如果用户输入的内荣在文档中并且相关，请根据相关的内容回答问题,
- 如果用户输入的内容和文档没有什么关系，回复在文档中没有相关的内容，并按正常的交流方式回答。
'''
tools = ['code_interpreter']  # `code_interpreter` is a built-in tool for executing code.
files = get_all_file('/work/home/onescience2025/deepseek/agent/qwenagent_demo/res')  # Give the bot files to read.
bot = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                function_list=tools,
                files=files)

# Step 4: Run the agent as a chatbot.
messages = []  # This stores the chat history.
histroy_len = 1
assert histroy_len > 0,"history lenth must be positive integer"
while True:
    # For example, enter the query "draw a dog and rotate it 90 degrees".
    query = input('\nuser query: ')
    # Append the user query to the chat history.
    messages.append({'role': 'user', 'content': query})
    response = []
    response_plain_text = ''
    print('bot response:')
    for response in bot.run(messages=messages,enable_thinking=False):
        # Streaming output.
        response_plain_text = typewriter_print(response, response_plain_text)
    # Append the bot responses to the chat history.
    messages.extend(response)
    messages = messages[:histroy_len-1]
