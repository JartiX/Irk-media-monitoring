"""
Вспомогательные функции
"""
import re
import hashlib
from datetime import datetime
from typing import Optional


def clean_text(text: str) -> str:
    """Очистить текст от лишних пробелов и символов"""
    if not text:
        return ""

    # Убираем множественные пробелы
    text = re.sub(r'\s+', ' ', text)
    # Убираем пробелы в начале и конце
    text = text.strip()
    # Убираем нулевые символы
    text = text.replace('\x00', '')

    return text


def extract_date(date_str: str, formats: Optional[list[str]] = None) -> Optional[datetime]:
    """Извлечь дату из строки"""
    if not date_str:
        return None

    if formats is None:
        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%d.%m.%Y %H:%M",
            "%d.%m.%Y",
            "%Y-%m-%d",
        ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue

    return None


def generate_hash(text: str) -> str:
    """Генерировать хэш для текста (для идентификации дубликатов)"""
    return hashlib.md5(text.encode()).hexdigest()


def truncate_text(text: str, max_length: int = 500) -> str:
    """Обрезать текст до максимальной длины"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def extract_urls(text: str) -> list[str]:
    """Извлечь URL из текста"""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


def is_spam(text: str) -> bool:
    """Проверить является ли текст спамом"""
    spam_patterns = [
        r'(?i)подпишись',
        r'(?i)переходи по ссылке',
        r'(?i)заработок',
        r'(?i)без вложений',
        r'(?i)пиши в лс',
        r'(?i)розыгрыш',
        r'(?i)выиграй',
        r'(?i)акция',
        r'(?i)скидка \d+%',
    ]

    for pattern in spam_patterns:
        if re.search(pattern, text):
            return True

    return False


def normalize_source_id(source_type: str, identifier: str) -> str:
    """Нормализовать ID источника"""
    return f"{source_type}_{identifier}".lower().replace(" ", "_")
