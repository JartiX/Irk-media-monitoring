"""
Фильтрация по ключевым словам
"""
from typing import Tuple

from loguru import logger

import config
import patterns

class KeywordFilter:
    """Фильтр контента по ключевым словам"""

    def __init__(self):
        # Компилируем регулярные выражения для быстрого поиска
        self.positive_pattern = patterns.TOURISM_HIGH_REGEX
        self.low_impact_pattern = patterns.TOURISM_LOW_REGEX
        self.negative_pattern = patterns.NEGATIVE_REGEX
        self.useful_pattern = patterns.USEFUL_REGEX
        self.useless_pattern = patterns.USELESS_REGEX
        # Политические паттерны для строгой фильтрации комментариев
        self.political_pattern = patterns.POLITICAL_REGEX
        # Паттерн для нецензурной лексики
        self.profanity_pattern = patterns.PROFANITY_REGEX
        self.whitelist_pattern = patterns.TOURISM_WHITELIST
        # Паттерн с запретными словами
        self.ban_pattern = patterns.BAN_REGEX
        # Гео паттерны с меньшим влиянием
        self.geo_pattern = patterns.GEO_REGEX

    def is_political(self, text):
        whitelist_matches = [m.group() for m in self.whitelist_pattern.finditer(text)]
        if whitelist_matches:
            logger.debug(
                f"Совпадения с вайтлистом ({whitelist_matches })")
            return False
        
        political_matches = [m.group() for m in self.political_pattern.finditer(text)]
        if political_matches:
            logger.debug(
                f"Связь с политикой ({political_matches })")
            return True
        
        return False


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

        # Проверяем запретные ключевые слова
        ban_matches = [m.group() for m in self.ban_pattern.finditer(text_lower)]

        if ban_matches:
            logger.debug(f"Отклонено: запретные ключевые слова ({ban_matches })")
            return False, -1

        # Проверяем негативные ключевые слова
        negative_matches = [m.group() for m in self.negative_pattern.finditer(text_lower)]

        # Если есть негативные совпадения и нет позитивных - скорее всего не туристический контент. Если позитивные совпадения есть, ML сделает окончательное решение
        if negative_matches and positive_matches == 0:
            logger.debug(f"Отклонено: негативные ключевые слова ({negative_matches })")
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
                         f"Low-impact: {low_impact_matches}, GEO: {geo_matches}, "
                         f"Negative: {negative_matches}")

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

    def check_profanity(self, text: str) -> bool:
        """
        Проверить наличие нецензурной лексики в тексте.

        Args:
            text: Текст для проверки

        Returns:
            bool: True если текст содержит нецензурную лексику
        """
        if not text:
            return False

        text_lower = text.lower()

        matches = [m.group() for m in self.profanity_pattern.finditer(text_lower)]
        if matches:
            logger.debug(f"Нецензурная лексика обнаружена в: {text[:50]} {matches}")
            return True

        return False

    def check_tourism_relevance(self, text: str) -> bool:
        """
        Проверить связь комментария с туризмом.

        Args:
            text: Текст для проверки

        Returns:
            bool: True если текст связан с туризмом
        """
        if not text or len(text) < config.PARSE_SETTINGS["min_comment_length"]:
            return False

        text_lower = text.lower()

        # Проверяем туристические ключевые слова
        tourism_matches = [m.group() for m in self.positive_pattern.finditer(text_lower)]
        geo_matches = [m.group() for m in self.geo_pattern.finditer(text_lower)]

        if tourism_matches:
            logger.debug(f"Tourism matches: {tourism_matches}")
        if geo_matches:
            logger.debug(f"Geo matches: {geo_matches}")

        # Релевантен если есть туристические слова или гео
        return bool(tourism_matches or geo_matches)

    def filter_comments(self, comments: list) -> list:
        """
        Отфильтровать список комментариев и проставить флаги.

        Устанавливает 4 независимых флага:
        - is_clean: без политики и без мата
        - is_relevant: связь с туризмом
        - is_political: содержит политику
        - is_profane: содержит нецензурную лексику

        Args:
            comments: Список комментариев

        Returns:
            list: Список комментариев с обновлёнными флагами
        """
        political_count = 0
        profane_count = 0
        relevant_count = 0
        clean_count = 0

        for comment in comments:
            text = comment.content
            logger.debug(f"Фильтрация комментария {text[:60]}")
            # Проверяем каждый флаг независимо
            is_political = self.is_political(text)
            is_profane = self.check_profanity(text)
            is_relevant = self.check_tourism_relevance(text)

            # Устанавливаем флаги
            comment.is_political = is_political
            comment.is_profane = is_profane
            comment.is_relevant = is_relevant
            comment.is_clean = not is_political and not is_profane

            # Считаем статистику
            if is_political:
                political_count += 1
            if is_profane:
                profane_count += 1
            if is_relevant:
                relevant_count += 1
            if comment.is_clean:
                clean_count += 1

        # Логируем статистику
        total = len(comments)
        if total > 0:
            logger.info(
                f"Фильтрация комментариев: всего={total}, "
                f"чистых={clean_count}, релевантных={relevant_count}, "
                f"политических={political_count}, с матом={profane_count}"
            )

        return comments
