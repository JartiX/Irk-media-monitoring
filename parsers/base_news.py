"""
Базовый класс для парсеров новостных сайтов
"""
from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Optional
import asyncio

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
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0"
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

    async def _fetch_page(self, url: str, session: aiohttp.ClientSession, max_retries: int = 3) -> Optional[BeautifulSoup]:
        """Загрузить страницу и вернуть BeautifulSoup объект с retry логикой"""
        last_error = None

        for attempt in range(max_retries):
            try:
                # Добавляем небольшую задержку перед повторными попытками
                if attempt > 0:
                    delay = 2 ** attempt  # Экспоненциальная задержка: 2, 4, 8 секунд
                    self.log_debug(f"Повторная попытка #{attempt + 1} после {delay}s задержки для {url}")
                    await asyncio.sleep(delay)

                async with session.get(url, headers=self.DEFAULT_HEADERS, timeout=30) as response:
                    if response.status == 200:
                        content = await response.text()
                        return BeautifulSoup(content, "lxml")
                    elif response.status in (403, 428, 429):  # Блокировка/Rate limit
                        self.log_debug(f"HTTP {response.status} для {url} (попытка {attempt + 1}/{max_retries})")
                        last_error = f"HTTP {response.status}"
                        if attempt < max_retries - 1:
                            continue  # Повторить попытку
                    else:
                        self.log_debug(f"HTTP {response.status} для {url}")
                        return None

            except asyncio.TimeoutError:
                last_error = "Timeout"
                self.log_error(f"Timeout при загрузке {url} (попытка {attempt + 1}/{max_retries})")
            except Exception as e:
                last_error = str(e)
                self.log_error(f"Ошибка загрузки {url}: {e} (попытка {attempt + 1}/{max_retries})")

        # Все попытки исчерпаны
        if last_error:
            self.log_error(f"Не удалось загрузить {url} после {max_retries} попыток. Последняя ошибка: {last_error}")
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
