import os
import re
import logging
import logging.handlers
from langchain_core.messages.base import get_msg_title_repr


def setup_global_logger(log_file_path, log_level=logging.INFO):
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=0,
        encoding="utf-8",
    )

    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def gen_job_log_name(job_name, directory):
    max_num = 0
    if not os.path.exists(directory):
        return f"{re.escape(job_name)}_{max_num + 1}.log"
    pattern = re.compile(rf'^{re.escape(job_name)}_(\d+)\.log$')
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)

    log_name = f"{re.escape(job_name)}_{max_num + 1}.log"
    return log_name


def is_interactive_env() -> bool:
    """Determine if running within IPython or Jupyter."""
    import sys

    return hasattr(sys, "ps2")


def pretty_print(message, printout=True):
    if isinstance(message, tuple):
        title = message
    elif isinstance(message.content, list):
        title = get_msg_title_repr(
            message.type.title().upper() + " Message", bold=is_interactive_env()
        )
        if message.name is not None:
            title += f"\nName: {message.name}"

        for i in message.content:
            if i["type"] == "text":
                title += f"\n{i['text']}\n"
            elif i["type"] == "tool_use":
                title += f"\nTool: {i['name']}"
                title += f"\nInput: {i['input']}"
        if printout:
            print(f"{title}")
    else:
        title = get_msg_title_repr(
            message.type.title() + " Message", bold=is_interactive_env()
        )
        if message.name is not None:
            title += f"\nName: {message.name}"
        title += f"\n\n{message.content}"
        if printout:
            print(f"{title}")
    return title
