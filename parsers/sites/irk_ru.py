"""
Парсер для сайта IRK.ru
"""
import re
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup

from ..base_news import BaseNewsParser
from database.models import Post
from utils.helpers import clean_text, extract_date, generate_hash
import config


class IrkRuParser(BaseNewsParser):
    """
    Парсер для сайта IRK.ru

    Особенности:
    - Парсит раздел /tourism/ (туристический контент)
    - Паттерн ссылок: /tourism/blog/ и /news/articles/
    - Контент: .j-article-main (исключая .j-hot-discussion, .j-similar)
    - Комментарии загружаются через JS, не парсятся
    """

    # Паттерн для поиска ссылок на статьи
    LINK_PATTERN = re.compile(r"(/tourism/blog/|/news/articles/)\d{8}/")

    # Селекторы для извлечения контента
    CONTENT_SELECTOR = ".j-article-main"
    JUNK_SELECTORS = [".j-hot-discussion", ".j-similar", ".comments", ".sidebar"]

    # Селекторы для заголовка
    TITLE_SELECTORS = ["h1", ".article-title", "meta[property='og:title']", "title"]

    def __init__(self, source_id: str):
        super().__init__(
            source_id=source_id,
            source_name="IRK.ru",
            base_url="https://www.irk.ru"
        )
        self.tourism_section = "/tourism/"
        # Контент из раздела туризма уже релевантен
        self.skip_relevance_check = True

    async def _parse_section(self) -> list[Post]:
        """Парсинг раздела туризма IRK.ru"""
        posts = []
        url = f"{self.base_url}{self.tourism_section}"

        async with aiohttp.ClientSession() as session:
            soup = await self._fetch_page(url, session)
            if not soup:
                return posts

            # Получаем ссылки на статьи
            links = await self._get_article_links(soup)
            self.log_info(f"Найдено {len(links)} ссылок на статьи")

            # Парсим каждую статью
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
        """Получить ссылки на статьи из раздела туризма"""
        links = []
        seen = set()

        # Ищем ссылки по паттерну
        for a_tag in soup.find_all("a", href=self.LINK_PATTERN):
            href = a_tag.get("href", "")
            link_text = a_tag.get_text().strip()

            # Пропускаем мусорные ссылки (комментарии, отзывы)
            if self._is_junk_link(link_text):
                continue

            if href and href not in seen:
                seen.add(href)
                links.append(href)

        return links

    async def _parse_article(self, soup: BeautifulSoup, url: str) -> Optional[Post]:
        """Парсинг статьи IRK.ru"""
        # Извлекаем заголовок
        title = self._extract_title(soup)
        if not title:
            self.log_debug(f"Не найден заголовок для {url}")
            return None

        # Извлекаем контент
        content = self._extract_content(soup)
        if not content:
            content = title

        # Извлекаем дату
        published_at = self._extract_date(soup)
        self.log_debug(f"Дата публикации: {published_at} для {url}")

        # Проверяем свежесть
        if not self._is_recent(published_at):
            self.log_debug(f"Статья слишком старая: {published_at} для {url}")
            return None

        # Извлекаем метрики
        comments_count = self._extract_comments_count(soup)
        views_count = self._extract_views_count(soup)

        return Post(
            source_id=self.source_id,
            external_id=generate_hash(url),
            title=title,
            content=content,
            url=url,
            published_at=published_at,
            comments_count=comments_count,
            views_count=views_count,
        )

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Извлечь заголовок статьи"""
        for selector in self.TITLE_SELECTORS:
            elem = soup.select_one(selector)
            if elem:
                # Для meta тегов берём content
                title = elem.get("content") or clean_text(elem.get_text())
                if title:
                    return title
        return ""

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Извлечь контент статьи"""
        content_elem = soup.select_one(self.CONTENT_SELECTOR)

        if content_elem:
            # Удаляем мусорные блоки
            for junk_selector in self.JUNK_SELECTORS:
                for junk in content_elem.select(junk_selector):
                    junk.decompose()

            return clean_text(content_elem.get_text())

        # Fallback: берём параграфы
        paragraphs = soup.select("p")
        return " ".join(clean_text(p.get_text()) for p in paragraphs[:10])

    def _extract_date(self, soup: BeautifulSoup) -> Optional:
        """Извлечь дату публикации"""
        date_selectors = [
            "time",
            ".date",
            ".published",
            "meta[property='article:published_time']"
        ]

        for selector in date_selectors:
            elem = soup.select_one(selector)
            if elem:
                date_str = elem.get("datetime") or elem.get("content") or elem.get_text()
                date = extract_date(date_str)
                if date:
                    return date

        return None

    def _extract_comments_count(self, soup: BeautifulSoup) -> int:
        """Извлечь количество комментариев"""
        selectors = [
            'a[href*="#comments"]',
            '.comments-count',
            '#comments-count'
        ]

        for selector in selectors:
            elem = soup.select_one(selector)
            if elem:
                match = re.search(r'(\d+)', elem.get_text())
                if match:
                    return int(match.group(1))

        return 0

    def _extract_views_count(self, soup: BeautifulSoup) -> int:
        """Извлечь количество просмотров"""
        selectors = ['.views', '.views-count', '.eye-count']

        for selector in selectors:
            elem = soup.select_one(selector)
            if elem:
                match = re.search(r'(\d+)', elem.get_text())
                if match:
                    return int(match.group(1))

        return 0

    def _is_junk_link(self, text: str) -> bool:
        """Проверить, является ли ссылка мусорной"""
        junk_pattern = re.compile(r'^(\d+\s*)?(отзыв|комментар|ответ)', re.IGNORECASE)
        return bool(junk_pattern.match(text))
