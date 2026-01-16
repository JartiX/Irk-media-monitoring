"""
Парсер для сайта Иркутскинформ (приангарье.рф / xn--h1aafalfhlffkls.xn--p1ai)
"""
import re
from typing import Optional
from datetime import datetime

import aiohttp
from bs4 import BeautifulSoup

from ..base_news import BaseNewsParser
from database.models import Post
from utils.helpers import clean_text, generate_hash


# Словарь русских месяцев
RUSSIAN_MONTHS = {
    'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4,
    'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8,
    'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12
}


def parse_russian_date(date_str: str) -> Optional[datetime]:
    """Парсинг даты в формате '22 декабря 2025'"""
    if not date_str:
        return None

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


class IrkutskoinformParser(BaseNewsParser):
    """
    Парсер для сайта приангарье.рф (Иркутскинформ)

    Особенности:
    - Домен: xn--h1aafalfhlffkls.xn--p1ai (Punycode для приангарье.рф)
    - URL статей: /news/{slug}/
    - Раздел туризма: /news/category/turizm/
    - Дата в формате: "22 декабря 2025"
    """

    def __init__(self, source_id: str):
        super().__init__(
            source_id=source_id,
            source_name="Иркутскинформ",
            base_url="https://xn--h1aafalfhlffkls.xn--p1ai"
        )
        self.tourism_section = "/news/category/turizm/"
        self.skip_relevance_check = False

    async def _parse_section(self) -> list[Post]:
        """Парсинг раздела туризма"""
        posts = []
        url = f"{self.base_url}{self.tourism_section}"

        async with aiohttp.ClientSession() as session:
            soup = await self._fetch_page(url, session)
            if not soup:
                self.log_error("Не удалось загрузить страницу раздела")
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

        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href', '').strip()

            # Убираем домен если есть
            if 'xn--h1aafalfhlffkls.xn--p1ai' in href:
                href = re.sub(r'https?://[^/]+', '', href)

            # Проверяем паттерн: /news/{slug}/ (исключая /news/category/)
            if self._is_article_link(href):
                if href not in seen:
                    seen.add(href)
                    links.append(href)

        return links

    def _is_article_link(self, href: str) -> bool:
        """Проверить, является ли ссылка статьей"""
        # Должен начинаться с /news/, но не /news/category/
        if not href.startswith('/news/'):
            return False
        if href.startswith('/news/category/'):
            return False

        # Убираем /news/ и проверяем что остался slug
        path = href[6:].strip('/')  # убираем /news/

        # Slug должен быть непустым и содержать только допустимые символы
        if not path or '/' in path:
            return False

        # Slug должен быть достаточно длинным
        if len(path) < 5:
            return False

        return True

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
        # h1
        h1 = soup.find('div', class_='h1')
        if h1:
            title = clean_text(h1.get_text())
            if title:
                return title
        return ""

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """
        Извлечь контент статьи из ck-content
        """
        content = soup.select_one('div.ck-content.news-inner')
        if not content:
            return ""

        paragraphs = [
            clean_text(p.get_text())
            for p in content.find_all('p')
            if p.get_text(strip=True)
        ]

        return " ".join(paragraphs)

    def _extract_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Извлечь дату публикации"""
        date_block = soup.select_one('.articleCardDate')
        if date_block:
            text = clean_text(date_block.get_text())
            date = parse_russian_date(text)
            if date:
                return date

        # Если не нашли, возвращаем None
        return None
