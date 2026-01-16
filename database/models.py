"""
Модели данных для системы мониторинга
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
from uuid import uuid4


@dataclass
class Source:
    """Источник данных (новостной сайт, группа ВК, Telegram канал)"""
    name: str
    type: str  # news, vk, telegram
    url: str
    is_active: bool = True
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data


@dataclass
class Post:
    """Пост или новость"""
    source_id: str
    external_id: str
    content: str
    url: str
    title: Optional[str] = None
    published_at: Optional[datetime] = None
    likes_count: int = 0
    views_count: int = 0
    comments_count: int = 0
    relevance_score: float = 0.0
    is_relevant: bool = False
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        if self.published_at:
            data["published_at"] = self.published_at.isoformat()
        return data


@dataclass
class Comment:
    """Комментарий к посту"""
    post_id: str
    external_id: str
    content: str
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    likes_count: int = 0
    is_useful: bool = False
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        if self.published_at:
            data["published_at"] = self.published_at.isoformat()
        return data
