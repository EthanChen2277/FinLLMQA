from typing import Any, Dict, Optional
from finllmqa.agent.autogen.logger.base_logger import BaseLogger
from finllmqa.agent.autogen.logger.sqlite_logger import SqliteLogger

__all__ = ("LoggerFactory",)


class LoggerFactory:
    @staticmethod
    def get_logger(logger_type: str = "sqlite", config: Optional[Dict[str, Any]] = None) -> BaseLogger:
        if config is None:
            config = {}

        if logger_type == "sqlite":
            return SqliteLogger(config)
        else:
            raise ValueError(f"[logger_factory] Unknown logger type: {logger_type}")
