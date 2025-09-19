import os
import logging
import logging.handlers
from langchain_core.messages.base import get_msg_title_repr


def setup_global_logger(log_file_path, log_level=logging.INFO):
    """
    配置全局日志记录器（根 logger）。
    所有模块的日志都会通过它写入指定文件。
    """
    log_dir = os.path.dirname(log_file_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # 获取根 logger
    root_logger = logging.getLogger()  # 不传参数即获取根 logger
    root_logger.setLevel(log_level)  # 设置根 logger 的最低日志级别

    # --- 创建 Handler ---
    # 1. 文件 Handler (写入文件，推荐使用 RotatingFileHandler)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=0,  # 保留 5 个备份文件
        encoding="utf-8",
    )

    # 2. (可选) 控制台 Handler
    console_handler = logging.StreamHandler()

    # --- 设置格式 ---
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # --- 将 Handler 添加到根 logger ---
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


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
