"""
Парсер ВКонтакте
"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional

import vk_api
from loguru import logger

from .base import BaseParser
from database.models import Post, Comment
from utils.helpers import clean_text, is_spam
import config


class VKParser(BaseParser):
    """Парсер для групп ВКонтакте"""

    def __init__(self, source_id: str, group_id: str, group_name: str):
        super().__init__(source_id, f"VK: {group_name}")
        self.group_id = group_id
        self.group_name = group_name
        self.days_lookback = config.PARSE_SETTINGS["days_lookback"]
        self.min_text_length = config.PARSE_SETTINGS["min_text_length"]
        self.min_comment_length = config.PARSE_SETTINGS["min_comment_length"]

        # Инициализация VK API
        if not config.VK_ACCESS_TOKEN:
            raise ValueError("VK_ACCESS_TOKEN не установлен")

        self.vk_session = vk_api.VkApi(token=config.VK_ACCESS_TOKEN)
        self.vk = self.vk_session.get_api()

    async def fetch_posts(self) -> list[Post]:
        """Получить посты из группы ВК"""
        posts = []
        cutoff_timestamp = int((datetime.utcnow() - timedelta(days=self.days_lookback)).timestamp())

        try:
            # Получаем ID группы если передан screen_name
            group_info = await self._get_group_info()
            if not group_info:
                return posts

            owner_id = -abs(group_info["id"])  # Отрицательный ID для групп

            # Получаем посты
            response = await asyncio.to_thread(
                self.vk.wall.get,
                owner_id=owner_id,
                count=self.max_posts,
                filter="owner"  # Только посты от имени группы
            )

            for item in response.get("items", []):
                try:
                    # Проверяем дату
                    post_date = item.get("date", 0)
                    if post_date < cutoff_timestamp:
                        continue

                    text = clean_text(item.get("text", ""))

                    # Пропускаем короткие посты
                    if len(text) < self.min_text_length:
                        continue

                    # Пропускаем репосты без текста
                    if not text and "copy_history" in item:
                        continue

                    post_id = item.get("id")
                    post_url = f"https://vk.com/wall{owner_id}_{post_id}"

                    post = Post(
                        source_id=self.source_id,
                        external_id=str(post_id),
                        title=None,
                        content=text,
                        url=post_url,
                        published_at=datetime.fromtimestamp(post_date),
                        likes_count=item.get("likes", {}).get("count", 0),
                        views_count=item.get("views", {}).get("count", 0),
                        comments_count=item.get("comments", {}).get("count", 0),
                    )
                    posts.append(post)

                except Exception as e:
                    self.log_debug(f"Ошибка парсинга поста: {e}")
                    continue

            self.log_info(f"Получено {len(posts)} постов")

        except vk_api.exceptions.ApiError as e:
            self.log_error(f"Ошибка VK API: {e}")
        except Exception as e:
            self.log_error("Ошибка при получении постов", e)

        return posts

    async def fetch_comments(self, post: Post) -> list[Comment]:
        """Получить комментарии к посту"""
        comments = []

        try:
            # Извлекаем owner_id из URL
            parts = post.url.replace("https://vk.com/wall", "").split("_")
            if len(parts) != 2:
                return comments

            owner_id = int(parts[0])
            post_id = int(parts[1])

            await self.delay()

            response = await asyncio.to_thread(
                self.vk.wall.getComments,
                owner_id=owner_id,
                post_id=post_id,
                count=100,
                sort="desc",
                extended=1
            )

            # Создаём словарь профилей для получения имён авторов
            profiles = {}
            for profile in response.get("profiles", []):
                profiles[profile["id"]] = f"{profile.get('first_name', '')} {profile.get('last_name', '')}"
            for group in response.get("groups", []):
                profiles[-group["id"]] = group.get("name", "")

            for item in response.get("items", []):
                try:
                    text = clean_text(item.get("text", ""))

                    # Пропускаем короткие комментарии
                    if len(text) < self.min_comment_length:
                        continue

                    # Пропускаем спам
                    if is_spam(text):
                        continue

                    from_id = item.get("from_id", 0)
                    author = profiles.get(from_id, "Аноним")

                    comment = Comment(
                        post_id=post.id,
                        external_id=str(item.get("id")),
                        content=text,
                        author=author,
                        published_at=datetime.fromtimestamp(item.get("date", 0)),
                        likes_count=item.get("likes", {}).get("count", 0),
                    )
                    comments.append(comment)

                except Exception as e:
                    self.log_debug(f"Ошибка парсинга комментария: {e}")
                    continue
            
            if comments:
                self.log_info(f"Получено {len(comments)} комментариев к посту {post.external_id}")

        except vk_api.exceptions.ApiError as e:
            if e.code == 15:  # Access denied
                self.log_debug("Комментарии закрыты")
            else:
                self.log_error(f"Ошибка VK API: {e}")
        except Exception as e:
            self.log_error("Ошибка при получении комментариев", e)

        return comments

    async def _get_group_info(self) -> Optional[dict]:
        """Получить информацию о группе"""
        try:
            response = await asyncio.to_thread(
                self.vk.groups.getById,
                group_id=self.group_id
            )
            if response:
                return response[0] if isinstance(response, list) else response.get("groups", [{}])[0]
        except vk_api.exceptions.ApiError as e:
            self.log_error(f"Ошибка получения информации о группе: {e}")
        return None
