from .base import BaseParser
from .base_news import BaseNewsParser
from .sites.irk_ru import IrkRuParser
from .vk_parser import VKParser
from .telegram_parser import TelegramParser

__all__ = [
    "BaseParser",
    "BaseNewsParser",
    "IrkRuParser",
    "VKParser",
    "TelegramParser",
]
