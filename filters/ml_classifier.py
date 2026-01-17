"""
ML классификатор релевантности на основе TF-IDF или BERT
"""
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


class MLClassifier:
    """
    ML классификатор с поддержкой TF-IDF и BERT backends.
    Выбор backend определяется настройкой classifier_type в config.ML_SETTINGS.
    """

    def __init__(self, classifier_type: str = None):
        """
        Инициализация классификатора.

        Args:
            classifier_type: Тип классификатора ("bert" или "tfidf").
                            Если None, используется значение из config.
        """
        self.classifier_type = classifier_type or config.ML_SETTINGS.get("classifier_type", "tfidf")
        self._backend = None
        self._initialize_backend()

    def _initialize_backend(self):
        """Инициализировать выбранный backend"""
        if self.classifier_type == "bert":
            try:
                from .bert_classifier import BertClassifier
                self._backend = BertClassifier()
                logger.info("Используется BERT классификатор")
            except ImportError as e:
                logger.warning(f"BERT недоступен (не установлены зависимости): {e}")
                logger.info("Переключение на TF-IDF классификатор")
                from .tfidf_classifier import TfidfClassifier
                self._backend = TfidfClassifier()
                self.classifier_type = "tfidf"
            except Exception as e:
                logger.warning(f"Ошибка инициализации BERT: {e}")
                logger.info("Переключение на TF-IDF классификатор")
                from .tfidf_classifier import TfidfClassifier
                self._backend = TfidfClassifier()
                self.classifier_type = "tfidf"
        else:
            from .tfidf_classifier import TfidfClassifier
            self._backend = TfidfClassifier()
            logger.info("Используется TF-IDF классификатор")

    @property
    def is_trained(self) -> bool:
        """Проверить, обучен ли классификатор"""
        return self._backend.is_trained if self._backend else False

    @property
    def threshold(self) -> float:
        """Получить порог релевантности"""
        return self._backend.threshold if self._backend else config.ML_SETTINGS["relevance_threshold"]

    def train(self, texts: list[str], labels: list[int]):
        """Обучить классификатор"""
        if self._backend:
            return self._backend.train(texts, labels)

    def predict(self, text: str) -> Tuple[bool, float]:
        """Предсказать релевантность текста"""
        if self._backend:
            return self._backend.predict(text)
        return False, 0.0

    def classify_posts(self, posts: list) -> list:
        """Классифицировать список постов"""
        if self._backend:
            return self._backend.classify_posts(posts)
        return posts

    @staticmethod
    def create_training_dataset() -> Tuple[list[str], list[int]]:
        """Создать датасет из примеров в ml/"""
        positive_examples = ml.POSITIVE_ML_TRAIN
        negative_examples = ml.NEGATIVE_ML_TRAIN

        texts = positive_examples + negative_examples
        labels = [1] * len(positive_examples) + [0] * len(negative_examples)

        return texts, labels

# Функция для инициализации и обучения модели
def initialize_classifier(classifier_type: str = None) -> MLClassifier:
    """
    Инициализировать и при необходимости обучить классификатор.

    Args:
        classifier_type: Тип классификатора ("bert" или "tfidf").
                        Если None, используется значение из config.

    Returns:
        MLClassifier: Инициализированный классификатор
    """
    classifier = MLClassifier(classifier_type=classifier_type)

    if not classifier.is_trained:
        logger.info("Обучаем ML классификатор на базовом датасете...")
        texts, labels = MLClassifier.create_training_dataset()
        classifier.train(texts, labels)

    return classifier
