"""
Базовый класс для парсеров новостных сайтов
"""
from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger

from .base import BaseParser
from database.models import Post, Comment
from utils.helpers import clean_text, extract_date, generate_hash
import config


class BaseNewsParser(BaseParser):
    """
    Базовый класс для всех новостных парсеров.

    Наследники должны реализовать:
    - _get_article_links(soup) - получение списка ссылок на статьи
    - _parse_article(soup, url) - парсинг конкретной статьи
    """

    # HTTP заголовки для запросов
    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    def __init__(self, source_id: str, source_name: str, base_url: str):
        super().__init__(source_id, source_name)
        self.base_url = base_url
        self.days_lookback = config.PARSE_SETTINGS["days_lookback"]
        self.skip_relevance_check = False

    async def fetch_posts(self) -> list[Post]:
        """Получить новости из источника"""
        posts = []

        try:
            posts = await self._parse_section()
            self.log_info(f"Получено {len(posts)} новостей")
        except Exception as e:
            self.log_error("Ошибка при получении новостей", e)

        return posts[:self.max_posts]

    async def fetch_comments(self, post: Post) -> list[Comment]:
        """Получить комментарии к новости"""
        # Большинство новостных сайтов не имеют открытых комментариев
        # или загружают их через JavaScript
        return []

    async def _fetch_page(self, url: str, session: aiohttp.ClientSession) -> Optional[BeautifulSoup]:
        """Загрузить страницу и вернуть BeautifulSoup объект"""
        try:
            async with session.get(url, headers=self.DEFAULT_HEADERS, timeout=30) as response:
                if response.status != 200:
                    self.log_debug(f"HTTP {response.status} для {url}")
                    return None

                content = await response.text()
                return BeautifulSoup(content, "lxml")
        except Exception as e:
            self.log_debug(f"Ошибка загрузки {url}: {e}")
            return None

    def _make_absolute_url(self, url: str) -> str:
        """Преобразовать относительный URL в абсолютный"""
        if url.startswith("http"):
            return url
        return f"{self.base_url}{url}"

    def _is_recent(self, published_at: Optional[datetime]) -> bool:
        """Проверить, что публикация не старше days_lookback дней"""
        if not published_at:
            return True  # Если дата неизвестна, считаем свежей

        cutoff_date = datetime.utcnow() - timedelta(days=self.days_lookback)
        return published_at >= cutoff_date

    @abstractmethod
    async def _parse_section(self) -> list[Post]:
        """
        Парсинг раздела сайта.
        Должен быть реализован в наследниках.
        """
        pass

    @abstractmethod
    async def _get_article_links(self, soup: BeautifulSoup) -> list[str]:
        """
        Получить список ссылок на статьи из страницы раздела.
        Должен быть реализован в наследниках.
        """
        pass

    @abstractmethod
    async def _parse_article(self, soup: BeautifulSoup, url: str) -> Optional[Post]:
        """
        Парсинг конкретной статьи.
        Должен быть реализован в наследниках.
        """
        pass
