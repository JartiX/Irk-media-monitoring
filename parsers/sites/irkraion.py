"""
Парсер для сайта Иркутского района (irkraion.ru)
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
    """Парсинг даты в формате '10 декабря 2025'"""
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


class IrkRaionParser(BaseNewsParser):
    """
    Парсер для сайта irkraion.ru (Иркутский район)

    Особенности:
    - Joomla сайт
    - URL статей: /news/turizm/{id}-{slug}
    - Контент: .item-page
    - Дата в формате: "10 декабря 2025"
    """

    def __init__(self, source_id: str):
        super().__init__(
            source_id=source_id,
            source_name="Иркутский район",
            base_url="https://www.irkraion.ru"
        )
        self.tourism_section = "/news/turizm"
        self.skip_relevance_check = True

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
            if href.startswith('https://www.irkraion.ru'):
                href = href.replace('https://www.irkraion.ru', '')
            elif href.startswith('http://www.irkraion.ru'):
                href = href.replace('http://www.irkraion.ru', '')

            # Проверяем паттерн: /news/turizm/{число}-{slug}
            if self._is_article_link(href):
                if href not in seen:
                    seen.add(href)
                    links.append(href)

        return links

    def _is_article_link(self, href: str) -> bool:
        """Проверить, является ли ссылка статьей"""
        # Паттерн: /news/turizm/14270-skolko-zim-...
        pattern = r'^/news/turizm/\d+-[a-z0-9-]+'
        return bool(re.match(pattern, href, re.IGNORECASE))

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
        # og:title
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return clean_text(og_title['content'])

        # h1
        h1 = soup.find('h1')
        if h1:
            title = clean_text(h1.get_text())
            if title:
                return title

        # title тег
        title_tag = soup.find('title')
        if title_tag:
            return clean_text(title_tag.get_text().split('|')[0])

        return ""

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Извлечь контент статьи"""
        # Удаляем мусорные элементы
        junk_patterns = [
            'script', 'style', 'nav', '.breadcrumbs', '.sidebar',
            '.mod_search', '#js-show-iframe-wrapper', 'footer',
            '.category-list', '.pagination', '.newsflash',
            '.article-info',  # Информация о материале
            '.item-info',     # Дополнительная инфо
            '.tags',          # Теги
            '.social',        # Соцсети
            '.share',         # Поделиться
            'header',         # Заголовки разделов
        ]
        for pattern in junk_patterns:
            for elem in soup.select(pattern):
                elem.decompose()

        # Ищем itemprop="articleBody" (основной контент)
        article_body = soup.find(itemprop='articleBody')
        if article_body:
            # Удаляем заголовки из контента (они уже в title)
            for h in article_body.find_all(['h1', 'h2', 'h3']):
                h.decompose()

            # Удаляем метаданные
            for elem in article_body.find_all(string=re.compile(r'(Информация о материале|Категория:|Опубликовано:|Просмотров:|Автор:|Подробности)', re.IGNORECASE)):
                try:
                    if elem.parent and hasattr(elem.parent, 'decompose'):
                        elem.parent.decompose()
                except (AttributeError, TypeError):
                    continue

            text = clean_text(article_body.get_text())
            if len(text) > 100:
                return text

        return ""

    def _extract_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Извлечь дату публикации"""
        # Ищем <time> с itemprop="datePublished"
        time_elem = soup.find('time', itemprop='datePublished')
        if time_elem:
            datetime_attr = time_elem.get('datetime')
            if datetime_attr:
                try:
                    date_str = datetime_attr.split('+')[0].split('Z')[0]
                    return datetime.fromisoformat(date_str)
                except ValueError:
                    pass
                
        return None
