"""
Парсер Telegram каналов
"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.types import Channel, Message
from loguru import logger

from .base import BaseParser
from database.models import Post, Comment
from utils.helpers import clean_text, is_spam
import config


class TelegramParser(BaseParser):
    """Парсер для Telegram каналов"""

    def __init__(self, source_id: str, channel_username: str):
        super().__init__(source_id, f"TG: {channel_username}")
        self.channel_username = channel_username
        self.days_lookback = config.PARSE_SETTINGS["days_lookback"]
        self.min_text_length = config.PARSE_SETTINGS["min_text_length"]
        self.min_comment_length = config.PARSE_SETTINGS["min_comment_length"]

        # Проверка настроек
        if not config.TELEGRAM_API_ID or not config.TELEGRAM_API_HASH:
            raise ValueError("TELEGRAM_API_ID и TELEGRAM_API_HASH должны быть установлены")

        self.api_id = int(config.TELEGRAM_API_ID)
        self.api_hash = config.TELEGRAM_API_HASH
        self.session_string = config.TELEGRAM_SESSION_STRING
        self.client: Optional[TelegramClient] = None

    async def _get_client(self) -> TelegramClient:
        """Получить или создать клиент Telegram"""
        if self.client is None:
            if self.session_string:
                session = StringSession(self.session_string)
            else:
                session = StringSession()

            self.client = TelegramClient(
                session,
                self.api_id,
                self.api_hash
            )

        if not self.client.is_connected():
            await self.client.start(phone=config.TELEGRAM_PHONE)

        return self.client

    async def fetch_posts(self) -> list[Post]:
        """Получить посты из Telegram канала"""
        posts = []
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.days_lookback)

        try:
            client = await self._get_client()

            # Получаем сущность канала
            try:
                entity = await client.get_entity(self.channel_username)
            except Exception as e:
                self.log_error(f"Канал не найден: {self.channel_username}", e)
                return posts

            if not isinstance(entity, Channel):
                self.log_error(f"{self.channel_username} не является каналом")
                return posts

            # Получаем сообщения
            async for message in client.iter_messages(
                entity,
                limit=self.max_posts,
                offset_date=datetime.now(timezone.utc)
            ):
                try:
                    # Проверяем дату
                    if message.date < cutoff_date:
                        break

                    # Пропускаем сообщения без текста
                    if not message.text:
                        continue

                    text = clean_text(message.text)

                    # Пропускаем короткие посты
                    if len(text) < self.min_text_length:
                        continue

                    post_url = f"https://t.me/{self.channel_username}/{message.id}"

                    # Подсчёт лайков через reactions
                    likes_count = 0
                    if message.reactions and message.reactions.results:
                        for reaction in message.reactions.results:
                            likes_count += reaction.count

                    post = Post(
                        source_id=self.source_id,
                        external_id=str(message.id),
                        title=None,
                        content=text,
                        url=post_url,
                        published_at=message.date.replace(tzinfo=None),
                        views_count=message.views or 0,
                        likes_count=likes_count,
                    )
                    posts.append(post)

                except Exception as e:
                    self.log_debug(f"Ошибка парсинга сообщения: {e}")
                    continue

            self.log_info(f"Получено {len(posts)} постов")

        except Exception as e:
            self.log_error("Ошибка при получении постов", e)

        return posts

    async def fetch_comments(self, post: Post) -> list[Comment]:
        """Получить комментарии к посту в Telegram"""
        comments = []

        try:
            client = await self._get_client()
            entity = await client.get_entity(self.channel_username)

            message_id = int(post.external_id)

            # Проверяем есть ли у поста комментарии (discussion)
            try:
                async for reply in client.iter_messages(
                    entity,
                    reply_to=message_id,
                    limit=50
                ):
                    try:
                        if not reply.text:
                            continue

                        text = clean_text(reply.text)

                        # Пропускаем короткие комментарии
                        if len(text) < self.min_comment_length:
                            continue

                        # Пропускаем спам
                        if is_spam(text):
                            continue

                        # Получаем имя автора
                        author = "Аноним"
                        if reply.sender:
                            if hasattr(reply.sender, "first_name"):
                                author = f"{reply.sender.first_name or ''} {reply.sender.last_name or ''}".strip()
                            elif hasattr(reply.sender, "title"):
                                author = reply.sender.title

                        # Подсчёт лайков комментария через reactions
                        comment_likes = 0
                        if hasattr(reply, 'reactions') and reply.reactions and reply.reactions.results:
                            for reaction in reply.reactions.results:
                                comment_likes += reaction.count

                        comment = Comment(
                            post_id=post.id,
                            external_id=str(reply.id),
                            content=text,
                            author=author,
                            published_at=reply.date.replace(tzinfo=None) if reply.date else None,
                            likes_count=comment_likes,
                        )
                        comments.append(comment)

                    except Exception as e:
                        self.log_debug(f"Ошибка парсинга комментария: {e}")
                        continue

            except Exception as e:
                # Канал может не иметь включенных комментариев
                self.log_debug(f"Комментарии недоступны: {e}")

            if comments:
                self.log_info(f"Получено {len(comments)} комментариев к посту {post.external_id}")

        except Exception as e:
            self.log_error("Ошибка при получении комментариев", e)

        return comments

    async def disconnect(self):
        """Отключиться от Telegram"""
        if self.client and self.client.is_connected():
            await self.client.disconnect()

    @staticmethod
    async def generate_session_string() -> str:
        """
        Генерация строки сессии для использования в GitHub Actions.
        """
        if not config.TELEGRAM_API_ID or not config.TELEGRAM_API_HASH:
            raise ValueError("Установите TELEGRAM_API_ID и TELEGRAM_API_HASH")

        client = TelegramClient(
            StringSession(),
            int(config.TELEGRAM_API_ID),
            config.TELEGRAM_API_HASH
        )

        await client.start(phone=config.TELEGRAM_PHONE)
        session_string = client.session.save()
        await client.disconnect()

        print(f"Ваша session string:\n{session_string}")
        print("\nДобавьте её в .env как TELEGRAM_SESSION_STRING")

        return session_string
