"""
Фильтрация по ключевым словам
"""
import re
from typing import Tuple

from loguru import logger

import config


class KeywordFilter:
    """Фильтр контента по ключевым словам"""

    def __init__(self):
        # Компилируем регулярные выражения для быстрого поиска
        self.positive_pattern = config.TOURISM_HIGH_REGEX
        self.low_impact_pattern = config.TOURISM_LOW_REGEX
        self.negative_pattern = config.NEGATIVE_REGEX
        self.useful_pattern = config.USEFUL_REGEX
        self.useless_pattern = config.USELESS_REGEX
        # Политические паттерны для строгой фильтрации комментариев
        self.political_pattern = config.POLITICAL_REGEX
        self.whitelist_pattern = config.TOURISM_WHITELIST
        # Гео паттерны с меньшим влиянием
        self.geo_pattern = config.GEO_REGEX

    def is_political(self, text):
        whitelist_matches = [m.group() for m in self.whitelist_pattern.finditer(text)]
        if whitelist_matches:
            logger.debug(
                f"Пропущено: совпадения с вайтлистом ({whitelist_matches })")
            return False
        
        political_matches = [m.group() for m in self.political_pattern.finditer(text)]
        if political_matches:
            logger.debug(
                f"Отклонено: связь с политикой ({political_matches })")
            return True


    def check_relevance(self, text: str) -> Tuple[bool, float]:
        """
        Проверить релевантность текста туризму.

        Returns:
            Tuple[bool, float]: (is_relevant, score)
            - is_relevant: True если текст релевантен туризму
            - score: оценка релевантности от 0 до 1
        """
        if not text:
            return False, 0.0

        text_lower = text.lower()

        # Подсчитываем совпадения с позитивными ключевыми словами
        positive_matches = 0
        matched_keywords = []

        geo_matches = [m.group() for m in self.geo_pattern.finditer(text_lower)]
        geo_bonus = min(0.05 * len(geo_matches), 0.3)  # максимум +0.3 к score
        
        low_impact_matches = [m.group() for m in self.low_impact_pattern.finditer(text_lower)]
        low_impact_bonus = min(len(low_impact_matches) * 0.05, 0.3)  # максимум +0.3

        for match in self.positive_pattern.finditer(text_lower):
            positive_matches += 1
            matched_keywords.append(match.group())

        # Проверяем негативные ключевые слова
        negative_matches = [m.group() for m in self.negative_pattern.finditer(text_lower)]

        # Если много негативных совпадений - скорее всего не туристический контент
        if len(negative_matches) >= 2:
            logger.debug(f"Отклонено: много негативных ключевых слов ({negative_matches })")
            return False, -1
        
        # Проверяем связь с политикой
        is_political = self.is_political(text_lower)

        if is_political:
            return False, -1

        # Расчёт оценки релевантности
        if positive_matches == 0:
            # Если гео или слабовлиящие слова есть, можно дать небольшой шанс
            score = geo_bonus + low_impact_bonus
            score = max(0.0, min(1.0, score))
            if score > 0:
                logger.debug(f"Релевантно ({score:.2f}): Нет high-impact слов, но есть GEO/Low-impact: GEO={geo_matches}, Low-impact={low_impact_matches}")
                return False, score  # не релевантно, но есть маленький шанс
            return False, 0.0

        # Базовая оценка от количества совпадений
        # 1 совпадение = 0.3, 2 = 0.5, 3+ = 0.7+
        base_score = min(0.3 + (positive_matches - 1) * 0.2, 1.0)

        # Штраф за негативные совпадения
        score = base_score - (len(negative_matches) * 0.3)

        # Надбавка за гео
        score += geo_bonus + low_impact_bonus

        score = max(0.0, min(1.0, score))

        is_relevant = score >= 0.3  # Минимальный порог

        if is_relevant:
            logger.debug(f"Релевантно ({score:.2f}): High-impact: {matched_keywords}, "
                         f"Low-impact: {low_impact_matches}, GEO: {geo_matches}")

        return is_relevant, score

    def filter_posts(self, posts: list) -> list:
        """
        Отфильтровать список постов.

        Returns:
            list: Список постов с обновлёнными полями is_relevant и relevance_score
        """
        for post in posts:
            logger.debug(f"Keywords фильтрация: Фильтрация поста {post.content[:100]}")
            # Объединяем заголовок и контент для анализа
            full_text = f"{post.title or ''} {post.content}".strip()
            is_relevant, score = self.check_relevance(full_text)

            post.is_relevant = is_relevant
            post.relevance_score = score

        relevant_count = sum(1 for p in posts if p.is_relevant)
        logger.info(f"Keyword фильтрация: {relevant_count}/{len(posts)} постов релевантны туризму")

        return posts

    def check_political_content(self, text: str) -> bool:
        """
        Проверить наличие политического контента в тексте.
        Строгая фильтрация: 1 совпадение = политический контент.

        Args:
            text: Текст для проверки

        Returns:
            bool: True если текст содержит политический контент
        """
        if not text:
            return False

        text_lower = text.lower()


        if self.political_pattern.search(text_lower):
            logger.debug(f"Политический контент обнаружен в: {text[:50]}...")
            return True

        return False

    def filter_comments(self, comments: list) -> list:
        """
        Отфильтровать список комментариев.

        Исключает комментарии с политическим контентом и отмечает полезные.

        Args:
            comments: Список комментариев

        Returns:
            list: Отфильтрованный список комментариев с обновлённым флагом is_useful
        """
        filtered_comments = []
        political_count = 0

        for comment in comments:
            # Проверяем на политический контент - полностью исключаем
            if self.check_political_content(comment.content):
                political_count += 1
                continue

            # Проверяем полезность
            comment.is_useful = self.check_comment_usefulness(comment.content)
            filtered_comments.append(comment)

        if political_count > 0:
            logger.info(f"Исключено политических комментариев: {political_count}")

        useful_count = sum(1 for c in filtered_comments if c.is_useful)
        if filtered_comments:
            logger.info(f"Полезных комментариев: {useful_count}/{len(filtered_comments)}")

        return filtered_comments

    def check_comment_usefulness(self, text: str) -> bool:
        """
        Проверить полезность комментария.

        Комментарий считается полезным если содержит:
        - Отзыв или рекомендацию
        - Вопрос о месте/услуге
        - Личный опыт
        - Связь с туризмом
        """
        if not text or len(text) < config.PARSE_SETTINGS["min_comment_length"]:
            return False
        
        text_lower = text.lower()

        useful_matches = len(self.useful_pattern.findall(text_lower))

        touristic_matches = len(self.positive_pattern.findall(text_lower))

        useless_matches = len(self.useless_pattern.findall(text_lower))

        negative_matches = len(self.negative_pattern.findall(text_lower))

        return bool((useful_matches or touristic_matches) and not (useless_matches or negative_matches))
