"""
Главный скрипт мониторинга медиапространства туризма Иркутской области

Запуск: python main.py
"""
import asyncio
import sys
from datetime import datetime

from loguru import logger

import config
from database.supabase_client import SupabaseClient
from database.models import Source

from parsers.vk_parser import VKParser
from parsers.telegram_parser import TelegramParser
from filters.keywords import KeywordFilter
from filters.ml_classifier import MLClassifier, initialize_classifier


# Настройка логирования
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/monitoring_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",
    rotation="1 day",
    retention="30 days"
)


class MediaMonitor:
    """Главный класс мониторинга медиапространства"""

    def __init__(self):
        self.db = SupabaseClient()
        self.keyword_filter = KeywordFilter()
        self.ml_classifier: MLClassifier = None
        self.stats = {
            "posts_processed": 0,      # всего обработано
            "posts_new": 0,             # новых добавлено
            "posts_updated": 0,         # обновлено существующих
            "posts_relevant": 0,        # релевантных среди обработанных
            "comments_processed": 0,    # всего обработано
            "comments_new": 0,          # новых добавлено
            "comments_updated": 0,      # обновлено существующих
            "comments_useful": 0,       # полезных среди обработанных
            "errors": 0,
        }

    async def run(self):
        """Запустить процесс мониторинга"""
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("Запуск мониторинга медиапространства туризма")
        logger.info("=" * 60)

        # Инициализация ML классификатора
        try:
            self.ml_classifier = initialize_classifier()
        except Exception as e:
            logger.warning(f"ML классификатор недоступен: {e}")

        # Обрабатываем источники
        await self._process_news_sources()
        await self._process_vk_sources()
        await self._process_telegram_sources()

        # Выводим статистику
        elapsed = (datetime.now() - start_time).total_seconds()
        self._print_stats(elapsed)

    async def _process_news_sources(self):
        """Обработать новостные источники"""
        logger.info("\nОбработка новостных источников...")



        for name, url, parser_class in config.NEWS_SOURCES:
            try:
                source = await self._get_or_create_source(
                    name=name,
                    source_type="news",
                    url=url
                )

                parser = parser_class(source["id"])
                posts = await parser.fetch_posts()

                if posts:
                    await self._process_posts(posts, parser)

            except Exception as e:
                logger.error(f"Ошибка обработки {name}: {e}")
                self.stats["errors"] += 1

    async def _process_vk_sources(self):
        """Обработать источники ВКонтакте"""
        if not config.VK_ACCESS_TOKEN:
            logger.warning("⚠️ VK_ACCESS_TOKEN не установлен, пропускаем ВК")
            return

        logger.info("\n📱 Обработка ВКонтакте...")

        for group_id in config.VK_GROUPS:
            try:
                # Получаем или создаём источник
                source = await self._get_or_create_source(
                    name=f"VK: {group_id}",
                    source_type="vk",
                    url=f"https://vk.com/{group_id}"
                )

                # Парсим посты
                parser = VKParser(source["id"], group_id, group_id)
                posts = await parser.fetch_posts()

                if posts:
                    await self._process_posts(posts, parser, fetch_comments=True)

            except Exception as e:
                logger.error(f"Ошибка обработки VK {group_id}: {e}")
                self.stats["errors"] += 1

    async def _process_telegram_sources(self):
        """Обработать Telegram каналы"""
        if not config.TELEGRAM_API_ID or not config.TELEGRAM_API_HASH:
            logger.warning("⚠️ Telegram API не настроен, пропускаем Telegram")
            return

        logger.info("\n✈️ Обработка Telegram...")

        for channel in config.TELEGRAM_CHANNELS:
            parser = None
            try:
                # Получаем или создаём источник
                source = await self._get_or_create_source(
                    name=f"TG: {channel}",
                    source_type="telegram",
                    url=f"https://t.me/{channel}"
                )

                # Парсим посты
                parser = TelegramParser(source["id"], channel)
                posts = await parser.fetch_posts()

                if posts:
                    await self._process_posts(posts, parser, fetch_comments=True)

            except Exception as e:
                logger.error(f"Ошибка обработки Telegram {channel}: {e}")
                self.stats["errors"] += 1
            finally:
                if parser:
                    await parser.disconnect()

    async def _process_posts(self, posts: list, parser, fetch_comments: bool = False):
        """Обработать посты: фильтрация и сохранение"""
        # Проверяем, нужно ли пропустить проверку релевантности
        skip_relevance = getattr(parser, 'skip_relevance_check', False)

        if skip_relevance:
            for post in posts:
                post.is_relevant = True
                post.relevance_score = 1.0
            logger.info(f"Пропуск проверки релевантности: {len(posts)} постов помечены как релевантные")
        else:
            # Фильтрация по ключевым словам
            posts = self.keyword_filter.filter_posts(posts)

            # ML классификация
            if self.ml_classifier:
                posts = self.ml_classifier.classify_posts(posts)

        # Подсчёт обработанных и релевантных
        self.stats["posts_processed"] += len(posts)
        self.stats["posts_relevant"] += sum(1 for p in posts if p.is_relevant)

        # Сохраняем с подробной статистикой
        new_count, updated_count = self._add_posts_with_stats(posts)
        self.stats["posts_new"] += new_count
        self.stats["posts_updated"] += updated_count

        # Получаем комментарии для релевантных постов
        if fetch_comments:
            relevant_posts = [p for p in posts if p.is_relevant]
            for post in relevant_posts:
                try:
                    comments = await parser.fetch_comments(post)

                    # Фильтруем комментарии (исключаем политические, отмечаем полезные)
                    filtered_comments = self.keyword_filter.filter_comments(comments)

                    # Получаем пост из БД для обновления comments_count
                    db_post = self.db.get_post_by_external_id(
                        post.source_id, post.external_id
                    )

                    if db_post:
                        # Устанавливаем post_id для комментариев
                        for comment in filtered_comments:
                            comment.post_id = db_post["id"]

                        # Подсчёт комментариев
                        self.stats["comments_processed"] += len(
                            filtered_comments)
                        self.stats["comments_useful"] += sum(
                            1 for c in filtered_comments if c.is_useful
                        )

                        new_count, updated_count = self._add_comments_with_stats(
                            filtered_comments)
                        self.stats["comments_new"] += new_count
                        self.stats["comments_updated"] += updated_count

                        # Обновляем comments_count у поста в БД
                        total_comments = len(self.db.get_comments(db_post["id"]))
                        self.db.update_post_comments_count(db_post["id"], total_comments)

                except Exception as e:
                    logger.debug(f"Ошибка получения комментариев: {e}")

    def _add_posts_with_stats(self, posts: list) -> tuple[int, int]:
        """Добавить посты и вернуть (новых, обновлённых)"""

        new_count = len(self.db.add_posts_batch(posts))
        updated_count = len(posts) - new_count

        return new_count, updated_count

    def _add_comments_with_stats(self, comments: list) -> tuple[int, int]:
        """Добавить комментарии и вернуть (новых, обновлённых)"""
        new_count = len(self.db.add_comments_batch(comments))
        updated_count = len(comments) - new_count

        return new_count, updated_count

    async def _get_or_create_source(self, name: str, source_type: str, url: str) -> dict:
        """Получить существующий источник или создать новый"""
        existing = self.db.get_source_by_url(url)
        if existing:
            return existing

        source = Source(name=name, type=source_type, url=url)
        return self.db.add_source(source)

    def _print_stats(self, elapsed_seconds: float):
        """Вывести статистику выполнения"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 СТАТИСТИКА ВЫПОЛНЕНИЯ")
        logger.info("=" * 60)
        logger.info(f"⏱️  Время выполнения: {elapsed_seconds:.1f} секунд")
        logger.info(f"\n📝 ПОСТЫ:")
        logger.info(f"   Обработано: {self.stats['posts_processed']}")
        logger.info(f"   ➕ Добавлено новых: {self.stats['posts_new']}")
        logger.info(f"   🔄 Обновлено: {self.stats['posts_updated']}")
        logger.info(f"   ✅ Релевантных: {self.stats['posts_relevant']}")

        logger.info(f"\n💬 КОММЕНТАРИИ:")
        logger.info(f"   Обработано: {self.stats['comments_processed']}")
        logger.info(f"   ➕ Добавлено новых: {self.stats['comments_new']}")
        logger.info(f"   🔄 Обновлено: {self.stats['comments_updated']}")
        logger.info(f"   👍 Полезных: {self.stats['comments_useful']}")

        if self.stats["errors"] > 0:
            logger.warning(f"⚠️  Ошибок: {self.stats['errors']}")
        logger.info("=" * 60)

        # Статистика из БД
        try:
            db_stats = self.db.get_stats()
            logger.info("\n📈 СТАТИСТИКА БАЗЫ ДАННЫХ")
            logger.info(f"   Всего источников: {db_stats['sources_count']}")
            logger.info(f"   Всего постов: {db_stats['posts_count']}")
            logger.info(
                f"   Релевантных постов: {db_stats['relevant_posts_count']}")
            logger.info(f"   Всего комментариев: {db_stats['comments_count']}")
            logger.info(
                f"   Полезных комментариев: {db_stats['useful_comments_count']}")
        except Exception as e:
            logger.debug(f"Не удалось получить статистику БД: {e}")


async def main():
    """Главная функция"""
    monitor = MediaMonitor()
    await monitor.run()


if __name__ == "__main__":
    asyncio.run(main())
