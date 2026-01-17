"""
Парсер для сайта Travel-Baikal.info
"""
import re
from typing import Optional
from datetime import datetime

import aiohttp
from bs4 import BeautifulSoup

from ..base_news import BaseNewsParser
from database.models import Post
from utils.helpers import clean_text, generate_hash


# Словарь русских месяцев для парсинга дат
RUSSIAN_MONTHS = {
    'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4,
    'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8,
    'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12
}


def parse_russian_date(date_str: str) -> Optional[datetime]:
    """Парсинг даты в формате '19 ноября 2025'"""
    if not date_str:
        return None

    # Паттерн: день месяц год
    pattern = r'(\d{1,2})\s+(\w+)\s+(\d{4})'
    match = re.search(pattern, date_str.lower())

    if match:
        day = int(match.group(1))
        month_name = match.group(2)
        year = int(match.group(3))

        month = RUSSIAN_MONTHS.get(month_name)
        if month:
            try:
                return datetime(year, month, day)
            except ValueError:
                return None

    return None


class TravelBaikalParser(BaseNewsParser):
    """
    Парсер для сайта Travel-Baikal.info

    Особенности:
    - WordPress сайт
    - URL статей: /news/[slug]/
    - Контент: .entry-content или article
    - Дата в формате: "19 ноября 2025"
    """

    # Паттерн для поиска ссылок на новости
    LINK_PATTERN = re.compile(r'/news/[^/]+/')

    # Селекторы
    JUNK_SELECTORS = [
        '.wp-block-button', '.swiper', 'script', 'style',
        '.cookie_note', 'nav', 'footer', '.menu', '.sidebar'
    ]

    def __init__(self, source_id: str):
        super().__init__(
            source_id=source_id,
            source_name="Travel-Baikal.info",
            base_url="https://travel-baikal.info"
        )
        self.news_section = "/news"

        self.skip_relevance_check = False

    async def _parse_section(self) -> list[Post]:
        """Парсинг раздела новостей"""
        posts = []
        url = f"{self.base_url}{self.news_section}"

        async with aiohttp.ClientSession() as session:
            soup = await self._fetch_page(url, session)
            if not soup:
                return posts

            links = await self._get_article_links(soup)
            self.log_info(f"Найдено {len(links)} ссылок на статьи")

            for link in links[:self.max_posts]:
                article_url = self._make_absolute_url(link)
                article_soup = await self._fetch_page(article_url, session)

                if article_soup:
                    post = await self._parse_article(article_soup, article_url)
                    if post:
                        posts.append(post)
                        await self.delay()

        return posts

    async def _get_article_links(self, soup: BeautifulSoup) -> list[str]:
        """Получить ссылки на статьи"""
        links = []
        seen = set()

        for a_tag in soup.find_all('a', href=self.LINK_PATTERN):
            href = a_tag.get('href', '')

            # Пропускаем главную страницу новостей
            if href == '/news/' or href == '/news':
                continue

            if href and href not in seen:
                seen.add(href)
                links.append(href)

        return links

    async def _parse_article(self, soup: BeautifulSoup, url: str) -> Optional[Post]:
        """Парсинг статьи"""
        title = self._extract_title(soup)
        if not title:
            self.log_debug(f"Не найден заголовок для {url}")
            return None

        published_at = self._extract_date(soup)

        content = self._extract_content(soup)
        if not content:
            content = title


        if not self._is_recent(published_at):
            self.log_debug(f"Статья слишком старая: {published_at}")
            return None

        return Post(
            source_id=self.source_id,
            external_id=generate_hash(url),
            title=title,
            content=content,
            url=url,
            published_at=published_at,
        )

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Извлечь заголовок"""
        elem = soup.select_one("h1")
        if elem:
            title = clean_text(elem.get_text())
            if title:
                return title
        return ""

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Извлечь контент статьи"""
        content_elem = soup.select_one(".post-content")
        if content_elem:
            # Удаляем мусор
            for junk_selector in self.JUNK_SELECTORS:
                for junk in content_elem.select(junk_selector):
                    junk.decompose()

            text = clean_text(content_elem.get_text())
            if text:
                return text

        # Fallback
        paragraphs = soup.select('p')
        return " ".join(clean_text(p.get_text()) for p in paragraphs[:10])

    def _extract_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Извлечь дату публикации"""
        elem = soup.select_one('.post-date')
        if elem:
            date_str = elem.get('datetime') or elem.get_text()
            date = parse_russian_date(date_str)
            if date:
                return date

        return None
