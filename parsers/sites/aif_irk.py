"""
Парсер для сайта АиФ Иркутск (irk.aif.ru)
"""
import re
import json
from typing import Optional
from datetime import datetime

import aiohttp
from bs4 import BeautifulSoup

from ..base_news import BaseNewsParser
from database.models import Post
from utils.helpers import clean_text, generate_hash


class AifIrkParser(BaseNewsParser):
    """
    Парсер для сайта irk.aif.ru (АиФ Иркутск)

    Особенности:
    - URL статей: /{category}/{article-slug}
    - Раздел туризма: /tag/turizm
    - Категории: society, turizm, money, politic, culture, sport, incidents, health
    - JSON-LD метаданные
    """

    # Категории статей на сайте
    CATEGORIES = ['society', 'turizm', 'money', 'politic', 'culture', 'sport', 'incidents', 'health']

    def __init__(self, source_id: str):
        super().__init__(
            source_id=source_id,
            source_name="АиФ Иркутск",
            base_url="https://irk.aif.ru"
        )
        self.tourism_section = "/tag/turizm"
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
        """Получить ссылки на статьи из контейнера article_list"""
        links = []
        seen = set()

        # Сначала пробуем найти контейнер со списком статей
        article_list = soup.find(class_='article_list') or soup.find(class_='content_list_js')

        # Если контейнер найден, ищем только в нем
        search_scope = article_list if article_list else soup

        for a_tag in search_scope.find_all('a', href=True):
            href = a_tag.get('href', '').strip()

            # Убираем домен если он есть
            if href.startswith('https://irk.aif.ru'):
                href = href.replace('https://irk.aif.ru', '')
            elif href.startswith('http://irk.aif.ru'):
                href = href.replace('http://irk.aif.ru', '')

            # Пропускаем если это не относительный путь
            if not href.startswith('/'):
                continue

            # Проверяем что это ссылка на статью
            if self._is_article_link(href):
                if href not in seen:
                    seen.add(href)
                    links.append(href)

        return links

    def _is_article_link(self, href: str) -> bool:
        """Проверить, является ли ссылка статьей"""
        # Убираем начальный и конечный слэш для анализа
        path = href.strip('/')
        parts = path.split('/')

        # Должно быть ровно 2 части: категория и slug
        if len(parts) != 2:
            return False

        category, slug = parts

        # Категория должна быть из списка
        if category not in self.CATEGORIES:
            return False

        # Исключаем страницы подкатегорий (persona, sorevnovaniya и т.д.)
        # Это не статьи, а страницы разделов
        excluded_slugs = [
            'persona',      # Персона
            'sorevnovaniya', # Соревнования
            'sobytiya',     # События
            'konkursy',     # Конкурсы
            'vystavki',     # Выставки
            'premera',      # Премьера
            'novosti',      # Новости (раздел)
        ]
        if slug in excluded_slugs:
            return False

        # Slug должен содержать буквы и быть достаточно длинным
        if len(slug) < 5 or not re.match(r'^[a-z0-9-_]+$', slug):
            return False

        return True

    async def _parse_article(self, soup: BeautifulSoup, url: str) -> Optional[Post]:
        """Парсинг статьи"""
        title = self._extract_title(soup)
        if not title:
            self.log_debug(f"Не найден заголовок для {url}")
            return None

        content = self._extract_content(soup)
        if not content:
            content = title

        published_at = self._extract_date(soup)

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
        # JSON-LD
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and data.get('headline'):
                    return clean_text(data['headline'])
            except (json.JSONDecodeError, TypeError):
                pass
        # h1
        h1 = soup.find('h1')
        if h1:
            title = clean_text(h1.get_text())
            if title:
                return title

        return ""

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Извлечь контент статьи"""
        # Удаляем мусор
        junk_patterns = [
            'script', 'style', 'nav', 'header', 'footer',
            '.vk_groups', '.comments', '.social', '.sidebar',
            '[class*="adfox"]', '[class*="yandex_rtb"]', '.related',
            # Блоки "Статья по теме"
            '.img_inject', '.inj_link_box', '.inj_name', '.inj_text',
            '.left_inj', '.right_inj', '.size2'
        ]
        for pattern in junk_patterns:
            for elem in soup.select(pattern):
                elem.decompose()

        content_elem = soup.select_one('.article_text')
        if content_elem:
            text = clean_text(content_elem.get_text())
            if len(text) > 100:
                return text

        # Fallback: собираем параграфы
        paragraphs = soup.select('p')
        texts = [clean_text(p.get_text()) for p in paragraphs if len(clean_text(p.get_text())) > 50]
        if texts:
            return " ".join(texts[:10])

        return ""

    def _extract_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Извлечь дату публикации с временем"""
        # Ищем в тексте формат "DD.MM.YYYY HH:MM"
        text = soup.get_text()[:2000]
        match = re.search(r'(\d{2})\.(\d{2})\.(\d{4})\s+(\d{1,2}):(\d{2})', text)
        if match:
            try:
                day = int(match.group(1))
                month = int(match.group(2))
                year = int(match.group(3))
                hour = int(match.group(4))
                minute = int(match.group(5))
                return datetime(year, month, day, hour, minute)
            except ValueError:
                pass

        # Формат 25.11.2025 (без времени)
        match = re.search(r'(\d{2})\.(\d{2})\.(\d{4})', text)
        if match:
            try:
                day = int(match.group(1))
                month = int(match.group(2))
                year = int(match.group(3))
                return datetime(year, month, day)
            except ValueError:
                pass
                
        return None
