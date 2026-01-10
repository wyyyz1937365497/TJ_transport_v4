"""日志模块 - 使用Rich和标准logging"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# 自定义主题
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold white on red",
        "debug": "dim cyan",
        "timestamp": "dim cyan",
    }
)

# 全局console
console = Console(theme=custom_theme)


def setup_logger(
    name: str = "tj_transport",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rich_handler: bool = True,
) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志器名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径（可选）
        rich_handler: 是否使用Rich格式化输出

    Returns:
        配置好的Logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    # Rich处理器（控制台输出）
    if rich_handler:
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,
        )
        rich_handler.setFormatter(
            logging.Formatter(
                fmt="%(message)s",
                datefmt="[%X]",
            )
        )
        logger.addHandler(rich_handler)

    # 文件处理器
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "tj_transport") -> logging.Logger:
    """获取已配置的日志记录器"""
    return logging.getLogger(name)


# 便捷函数
def info(msg: str, **kwargs):
    """记录INFO级别日志"""
    get_logger().info(msg, **kwargs)


def debug(msg: str, **kwargs):
    """记录DEBUG级别日志"""
    get_logger().debug(msg, **kwargs)


def warning(msg: str, **kwargs):
    """记录WARNING级别日志"""
    get_logger().warning(msg, **kwargs)


def error(msg: str, **kwargs):
    """记录ERROR级别日志"""
    get_logger().error(msg, **kwargs)


def critical(msg: str, **kwargs):
    """记录CRITICAL级别日志"""
    get_logger().critical(msg, **kwargs)
