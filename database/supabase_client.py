"""
Клиент для работы с Supabase
"""
from typing import Optional
from loguru import logger
from supabase import create_client, Client

import config
from .models import Source, Post, Comment


class SupabaseClient:
    """Клиент для взаимодействия с базой данных Supabase"""

    def __init__(self):
        if not config.SUPABASE_URL or not config.SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_URL и SUPABASE_KEY должны быть установлены")

        self.client: Client = create_client(
            config.SUPABASE_URL,
            config.SUPABASE_KEY
        )
        logger.info("Подключение к Supabase установлено")

    # === Источники ===

    def get_sources(self, source_type: Optional[str] = None, active_only: bool = True) -> list[dict]:
        """Получить список источников"""
        query = self.client.table("sources").select("*")

        if source_type:
            query = query.eq("type", source_type)
        if active_only:
            query = query.eq("is_active", True)

        response = query.execute()
        return response.data

    def add_source(self, source: Source) -> dict:
        """Добавить новый источник"""
        response = self.client.table("sources").insert(
            source.to_dict()).execute()
        logger.info(f"Добавлен источник: {source.name}")
        return response.data[0]

    def get_source_by_url(self, url: str) -> Optional[dict]:
        """Найти источник по URL"""
        response = self.client.table("sources").select(
            "*").eq("url", url).execute()
        return response.data[0] if response.data else None

    # === Посты ===

    def post_exists(self, source_id: str, external_id: str) -> bool:
        """Проверить существует ли пост"""
        response = self.client.table("posts").select("id").eq(
            "source_id", source_id
        ).eq("external_id", external_id).execute()
        return len(response.data) > 0

    def add_post(self, post: Post, update_existing: bool = True) -> Optional[dict]:
        """Добавить новый пост или обновить существующий (upsert)

        Args:
            post: Пост для добавления/обновления
            update_existing: Если True, обновляет существующий пост

        Returns:
            dict с данными поста если добавлен новый, None если обновлён существующий
        """
        existing = self.get_post_by_external_id(post.source_id, post.external_id)

        if existing:
            if not update_existing:
                logger.debug(f"Пост уже существует: {post.external_id}")
                return None

            # Обновляем существующий пост
            update_data = {
                "title": post.title,
                "content": post.content,
                "likes_count": post.likes_count,
                "views_count": post.views_count,
                "comments_count": post.comments_count,
                "relevance_score": post.relevance_score,
                "is_relevant": post.is_relevant,
            }
            if post.published_at:
                update_data["published_at"] = post.published_at.isoformat()

            self.client.table("posts").update(update_data).eq("id", existing["id"]).execute()
            logger.debug(f"Обновлён пост: {post.external_id}")
            return None  # None = не новый пост

        response = self.client.table("posts").insert(post.to_dict()).execute()
        logger.debug(f"Добавлен пост: {post.external_id}")
        return response.data[0]

    def add_posts_batch(self, posts: list[Post]) -> list[dict]:
        """Добавить несколько постов (пакетная вставка)"""
        added = []
        for post in posts:
            result = self.add_post(post)
            if result:
                added.append(result)
        return added

    def get_posts(
        self,
        source_id: Optional[str] = None,
        relevant_only: bool = False,
        limit: int = 100
    ) -> list[dict]:
        """Получить посты"""
        query = self.client.table("posts").select("*")

        if source_id:
            query = query.eq("source_id", source_id)
        if relevant_only:
            query = query.eq("is_relevant", True)

        response = query.order("created_at", desc=True).limit(limit).execute()
        return response.data

    def update_post_relevance(self, post_id: str, is_relevant: bool, score: float):
        """Обновить релевантность поста"""
        self.client.table("posts").update({
            "is_relevant": is_relevant,
            "relevance_score": score
        }).eq("id", post_id).execute()

    def update_post_comments_count(self, post_id: str, comments_count: int):
        """Обновить количество комментариев у поста"""
        self.client.table("posts").update({
            "comments_count": comments_count
        }).eq("id", post_id).execute()

    def get_post_by_external_id(self, source_id: str, external_id: str) -> Optional[dict]:
        """Получить пост по внешнему ID"""
        response = self.client.table("posts").select("*").eq(
            "source_id", source_id
        ).eq("external_id", external_id).execute()
        return response.data[0] if response.data else None

    # === Комментарии ===

    def comment_exists(self, post_id: str, external_id: str) -> bool:
        """Проверить существует ли комментарий"""
        response = self.client.table("comments").select("id").eq(
            "post_id", post_id
        ).eq("external_id", external_id).execute()
        return len(response.data) > 0

    def get_comment_by_external_id(self, post_id: str, external_id: str) -> Optional[dict]:
        """Получить комментарий по внешнему ID"""
        response = self.client.table("comments").select("*").eq(
            "post_id", post_id
        ).eq("external_id", external_id).execute()
        return response.data[0] if response.data else None

    def add_comment(self, comment: Comment, update_existing: bool = True) -> Optional[dict]:
        """Добавить комментарий или обновить существующий (upsert)"""
        existing = self.get_comment_by_external_id(comment.post_id, comment.external_id)

        if existing:
            if not update_existing:
                logger.debug(f"Комментарий уже существует: {comment.external_id}")
                return None

            # Обновляем существующий комментарий
            update_data = {
                "content": comment.content,
                "author": comment.author,
                "likes_count": comment.likes_count,
                "is_useful": comment.is_useful,
            }
            if comment.published_at:
                update_data["published_at"] = comment.published_at.isoformat()

            self.client.table("comments").update(update_data).eq("id", existing["id"]).execute()
            logger.debug(f"Обновлён комментарий: {comment.external_id}")
            return None

        response = self.client.table("comments").insert(
            comment.to_dict()).execute()
        logger.info(f"Добавлен комментарий: {comment.external_id}")
        return response.data[0]

    def add_comments_batch(self, comments: list[Comment]) -> list[dict]:
        """Добавить несколько комментариев"""
        added = []
        for comment in comments:
            result = self.add_comment(comment)
            if result:
                added.append(result)
        return added

    def get_comments(self, post_id: str, useful_only: bool = False) -> list[dict]:
        """Получить комментарии к посту"""
        query = self.client.table("comments").select(
            "*").eq("post_id", post_id)

        if useful_only:
            query = query.eq("is_useful", True)

        response = query.order("published_at", desc=True).execute()
        return response.data

    def update_comment_usefulness(self, comment_id: str, is_useful: bool):
        """Обновить полезность комментария"""
        self.client.table("comments").update({
            "is_useful": is_useful
        }).eq("id", comment_id).execute()

    # === Статистика ===

    def get_stats(self) -> dict:
        """Получить статистику по базе данных"""
        sources = self.client.table("sources").select(
            "id", count="exact").execute()
        posts = self.client.table("posts").select(
            "id", count="exact").execute()
        relevant_posts = self.client.table("posts").select(
            "id", count="exact").eq("is_relevant", True).execute()
        comments = self.client.table("comments").select(
            "id", count="exact").execute()
        useful_comments = self.client.table("comments").select(
            "id", count="exact").eq("is_useful", True).execute()

        return {
            "sources_count": sources.count,
            "posts_count": posts.count,
            "relevant_posts_count": relevant_posts.count,
            "comments_count": comments.count,
            "useful_comments_count": useful_comments.count,
        }
