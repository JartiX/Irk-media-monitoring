"""
Базовый класс для всех парсеров
"""
from abc import ABC, abstractmethod
from typing import Optional
import asyncio

from loguru import logger

from database.models import Post, Comment
import config


class BaseParser(ABC):
    """Абстрактный базовый класс для парсеров"""

    def __init__(self, source_id: str, source_name: str):
        self.source_id = source_id
        self.source_name = source_name
        self.request_delay = config.PARSE_SETTINGS["request_delay"]
        self.max_posts = config.PARSE_SETTINGS["max_posts_per_source"]

    @abstractmethod
    async def fetch_posts(self) -> list[Post]:
        """Получить посты из источника"""
        pass

    @abstractmethod
    async def fetch_comments(self, post: Post) -> list[Comment]:
        """Получить комментарии к посту"""
        pass

    async def delay(self):
        """Задержка между запросами для предотвращения блокировки"""
        await asyncio.sleep(self.request_delay)

    def log_info(self, message: str):
        """Логирование информации"""
        logger.info(f"[{self.source_name}] {message}")

    def log_error(self, message: str, error: Optional[Exception] = None):
        """Логирование ошибок"""
        if error:
            logger.error(f"[{self.source_name}] {message}: {error}")
        else:
            logger.error(f"[{self.source_name}] {message}")

    def log_debug(self, message: str):
        """Логирование отладки"""
        logger.debug(f"[{self.source_name}] {message}")
