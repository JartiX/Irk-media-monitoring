import os
from typing import Tuple, Optional

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from loguru import logger

import config
import ml

class TfidfClassifier:
    """ML классификатор релевантности туристического контента"""

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[LogisticRegression] = None
        self.threshold = config.ML_SETTINGS["relevance_threshold"]

        self.is_trained = False

        # Пытаемся загрузить обученную модель
        self._load_model()

    def _load_model(self):
        """Загрузить обученную модель если существует"""
        model_path = config.ML_SETTINGS["tfidf_model_path"]
        vectorizer_path = config.ML_SETTINGS["tfidf_vectorizer_path"]

        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            try:
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                self.is_trained = True
                logger.info("ML модель загружена")
            except Exception as e:
                logger.warning(f"Ошибка загрузки модели: {e}")
                self.is_trained = False

    def train(self, texts: list[str], labels: list[int]):
        """
        Обучить классификатор.

        Args:
            texts: Список текстов для обучения
            labels: Метки (1 = релевантно туризму, 0 = нет)
        """
        if len(texts) < 50:
            logger.warning("Недостаточно данных для обучения (минимум 50)")
            return

        logger.info(f"Обучение на {len(texts)} примерах...")

        # Разделяем на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # TF-IDF векторизация
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Обучение модели
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced'
        )
        self.model.fit(X_train_vec, y_train)

        # Оценка
        y_pred = self.model.predict(X_test_vec)
        logger.info("\nОценка модели:")
        logger.info(classification_report(y_test, y_pred,
                    target_names=['Не туризм', 'Туризм']))

        # Сохраняем модель
        self._save_model()
        self.is_trained = True

    def _save_model(self):
        """Сохранить модель"""
        model_dir = os.path.dirname(config.ML_SETTINGS["tfidf_model_path"])
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(self.model, config.ML_SETTINGS["tfidf_model_path"])
        joblib.dump(self.vectorizer, config.ML_SETTINGS["tfidf_vectorizer_path"])
        logger.info("ML модель сохранена")

    def predict(self, text: str) -> Tuple[bool, float]:
        """
        Предсказать релевантность текста.

        Returns:
            Tuple[bool, float]: (is_relevant, probability)
        """
        if not self.is_trained or not text:
            return False, 0.0

        try:
            X = self.vectorizer.transform([text])
            proba = self.model.predict_proba(X)[0]

            # Вероятность класса "туризм" (индекс 1)
            tourism_proba = proba[1] if len(proba) > 1 else proba[0]

            is_relevant = tourism_proba >= self.threshold
            return is_relevant, float(tourism_proba)

        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return False, 0.0

    def classify_posts(self, posts: list) -> list:
        """
        Классифицировать список постов с помощью ML.

        Если модель не обучена, возвращает посты без изменений.
        """
        if not self.is_trained:
            logger.warning("ML модель не обучена, пропускаем классификацию")
            return posts

        for post in posts:
            logger.debug(
                f"ML классификация: Фильтрация поста {post.content[:100]}")

            full_text = f"{post.title or ''} {post.content}".strip()
            is_relevant, score = self.predict(full_text)

            # ML дополняет фильтрацию по ключевым словам
            # Если keywords уже пометили как релевантное, ML может подтвердить или опровергнуть

            # Комбинируем оценки: keyword_score * 0.4 + ml_score * 0.6
            combined_score = post.relevance_score * 0.4 + score * 0.6
            if post.is_relevant:

                # Если модель тоже считает пост релевантным, пропускаем
                if is_relevant:
                    post.is_relevant = True

                else:
                    post.is_relevant = combined_score >= self.threshold

                    # Логируем какие посты модель отклонила
                    if not post.is_relevant:
                        logger.debug(
                            f"Модель опровергла пост, который keyword посчитал релевантным")

                post.relevance_score = combined_score
            else:
                # Если keywords не нашёл, но ML уверен - помечаем как релевантное
                if score >= 0.7 and post.relevance_score >= 0:  # Высокий порог для ML-only и отсутствие негативных слов
                    post.is_relevant = True
                    post.relevance_score = score
                    logger.debug(f"Модель уверена в релевантности поста")
                else:
                    if post.relevance_score >= 0:
                        post.relevance_score = combined_score
            logger.debug(
                f"Итоговый статус - Релевантность:{post.is_relevant} Score:({post.relevance_score})")

        relevant_count = sum(1 for p in posts if p.is_relevant)
        logger.info(
            f"ML классификация: {relevant_count}/{len(posts)} постов релевантны туризму")

        return posts
