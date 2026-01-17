"""
BERT классификатор релевантности туристического контента
на базе cointegrated/rubert-tiny2
"""
import os
import shutil
from typing import Tuple, Optional, List
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from loguru import logger
import numpy as np

import config
import ml


class TourismDataset(Dataset):
    """Dataset для обучения BERT классификатора"""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BertClassifier:
    """
    BERT классификатор релевантности туристического контента.
    Базовая модель: cointegrated/rubert-tiny2
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация классификатора.

        Args:
            model_path: Путь к сохраненной модели. Если None, используется путь из config.
        """
        self.model_path = model_path or config.ML_SETTINGS.get("bert_model_path", "JartiX/bert_tourism_classifier")
        self.base_model = config.ML_SETTINGS.get("bert_base_model", "cointegrated/rubert-tiny2")
        self.max_length = config.ML_SETTINGS.get("bert_max_length", 256)
        self.device = self._get_device()
        self.threshold = config.ML_SETTINGS.get("relevance_threshold", 0.5)

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.is_trained = False

        # Пытаемся загрузить обученную модель
        self._load_model()

    def _get_device(self) -> torch.device:
        """Автоопределение устройства (GPU/CPU)"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"BERT: Используется GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("BERT: Используется Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            logger.info("BERT: GPU не найден, используется CPU")
        return device

    def _load_model(self) -> bool:
        """Загрузить обученную модель, если существует"""
        try:
            logger.info(f"Загрузка BERT модели: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

            self.model.to(self.device)
            self.model.eval()
            self.is_trained = True

            logger.info("BERT модель успешно загружена")
            return True
        
        except Exception as e:
            logger.warning(f"Ошибка загрузки BERT модели: {e}")
            self.is_trained = False
            return False

    def _initialize_base_model(self):
        """Инициализировать базовую модель для обучения"""
        logger.info(f"Загрузка базовой модели {self.base_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=2,
            problem_type="single_label_classification"
        )
        self.model.to(self.device)

    def train(
        self,
        texts: List[str],
        labels: List[int],
        epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        warmup_ratio: float = 0.1,
        max_length: int = None,
        validation_split: float = 0.2,
    ) -> dict:
        """
        Обучить классификатор с fine-tuning.

        Args:
            texts: Список текстов для обучения
            labels: Метки (1 = релевантно туризму, 0 = нет)
            epochs: Количество эпох обучения
            batch_size: Размер батча
            learning_rate: Скорость обучения
            warmup_ratio: Доля шагов для warmup
            max_length: Максимальная длина токенизации
            validation_split: Доля данных для валидации

        Returns:
            dict с метриками обучения
        """
        tmp_dir = Path("tmp/bert_train")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Используем значения из конфига если не переданы
        epochs = epochs or config.ML_SETTINGS.get("bert_epochs", 3)
        batch_size = batch_size or config.ML_SETTINGS.get("bert_batch_size", 16)
        learning_rate = learning_rate or config.ML_SETTINGS.get("bert_learning_rate", 2e-5)
        max_length = max_length or self.max_length

        if len(texts) < 50:
            logger.warning("Недостаточно данных для обучения (минимум 50)")
            return {"error": "insufficient_data"}

        logger.info(f"Начало обучения BERT на {len(texts)} примерах...")
        logger.info(f"Устройство: {self.device}, Epochs: {epochs}, Batch size: {batch_size}")

        # Инициализируем базовую модель
        self._initialize_base_model()

        # Разделяем данные
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels,
            test_size=validation_split,
            random_state=42,
            stratify=labels
        )

        logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}")

        # Создаем датасеты
        train_dataset = TourismDataset(X_train, y_train, self.tokenizer, max_length)
        val_dataset = TourismDataset(X_val, y_val, self.tokenizer, max_length)

        # Настройки обучения
        training_args = TrainingArguments(
            output_dir=str(tmp_dir / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_ratio=warmup_ratio,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=str(tmp_dir / "logs"),
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),  # Mixed precision на GPU
            report_to="none",  # Отключаем wandb и др.
        )

        # Функция для вычисления метрик
        def compute_metrics(eval_pred):
            predictions, labels_batch = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {
                "accuracy": accuracy_score(labels_batch, predictions),
                "f1": f1_score(labels_batch, predictions, average="binary"),
            }

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # Обучение
        logger.info("Запуск обучения...")
        train_result = trainer.train()

        # Оценка на валидации
        eval_result = trainer.evaluate()
        logger.info(f"Результаты валидации: accuracy={eval_result.get('eval_accuracy', 0):.4f}, f1={eval_result.get('eval_f1', 0):.4f}")

        # Детальный отчет
        self.model.eval()
        val_predictions = self._predict_batch_internal(X_val)
        val_pred_labels = [1 if p >= self.threshold else 0 for p in val_predictions]

        logger.info("\nОценка модели на валидационной выборке:")
        report = classification_report(
            y_val, val_pred_labels,
            target_names=['Не туризм', 'Туризм']
        )
        for line in report.split('\n'):
            if line.strip():
                logger.info(line)

        # Сохраняем модель
        self._save_model()
        self.is_trained = True

        metrics = {
            "train_loss": train_result.training_loss,
            "eval_accuracy": eval_result.get("eval_accuracy", 0),
            "eval_f1": eval_result.get("eval_f1", 0),
            "epochs": epochs,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }

        logger.info(f"Обучение завершено. F1: {metrics['eval_f1']:.4f}")

        shutil.rmtree(tmp_dir, ignore_errors=True)
        
        return metrics

    def _save_model(self):
        """Сохранить модель в формате transformers (safetensors)"""
        repo_id = self.model_path

        logger.info(f"Загружаем модель BERT на HuggingFace Hub: {repo_id}")

        self.model.push_to_hub(
            repo_id,
            safe_serialization=True,
        )
        self.tokenizer.push_to_hub(repo_id)

        logger.info("BERT модель успешно загружена в HuggingFace Hub")

    def predict(self, text: str) -> Tuple[bool, float]:
        """
        Предсказать релевантность текста.

        Args:
            text: Текст для классификации

        Returns:
            Tuple[bool, float]: (is_relevant, probability)
        """
        if not self.is_trained or not text:
            return False, 0.0

        try:
            proba = self._predict_single(text)
            is_relevant = proba >= self.threshold
            return is_relevant, float(proba)
        except Exception as e:
            logger.error(f"Ошибка предсказания BERT: {e}")
            return False, 0.0

    def _predict_single(self, text: str) -> float:
        """Внутренний метод для предсказания одного текста"""
        self.model.eval()

        with torch.no_grad():
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)

            # Вероятность класса "туризм" (индекс 1)
            return probs[0][1].item()

    def predict_batch(self, texts: List[str], batch_size: int = None) -> List[Tuple[bool, float]]:
        """
        Батчевое предсказание для списка текстов.

        Args:
            texts: Список текстов
            batch_size: Размер батча

        Returns:
            Список tuple (is_relevant, probability)
        """
        if not self.is_trained or not texts:
            return [(False, 0.0)] * len(texts)

        batch_size = batch_size or config.ML_SETTINGS.get("bert_batch_size", 16)

        try:
            probas = self._predict_batch_internal(texts, batch_size)
            return [(p >= self.threshold, float(p)) for p in probas]
        except Exception as e:
            logger.error(f"Ошибка батчевого предсказания BERT: {e}")
            return [(False, 0.0)] * len(texts)

    def _predict_batch_internal(self, texts: List[str], batch_size: int = 16) -> List[float]:
        """Внутренний метод для батчевого предсказания"""
        self.model.eval()
        all_probas = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            with torch.no_grad():
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)

                # Вероятности класса "туризм"
                batch_probas = probs[:, 1].cpu().numpy().tolist()
                all_probas.extend(batch_probas)

        return all_probas

    def classify_posts(self, posts: list) -> list:
        """
        Классифицировать список постов с помощью BERT.
        Совместимо с API MLClassifier.

        Args:
            posts: Список объектов Post

        Returns:
            Список постов с обновленными relevance_score и is_relevant
        """
        if not self.is_trained:
            logger.warning("BERT модель не обучена, пропускаем классификацию")
            return posts

        # Подготавливаем тексты
        texts = [f"{p.title or ''} {p.content}".strip() for p in posts]

        # Батчевое предсказание
        predictions = self.predict_batch(texts)

        # Обновляем посты
        for post, (is_relevant, score) in zip(posts, predictions):
            logger.debug(f"BERT классификация: Фильтрация поста {post.content[:100]}")

            # Комбинируем с существующей оценкой от keyword filter
            # keyword_score * 0.4 + ml_score * 0.6
            combined_score = post.relevance_score * 0.4 + score * 0.6

            if post.is_relevant:
                # Если keyword filter считает релевантным
                if is_relevant:
                    post.is_relevant = True
                else:
                    post.is_relevant = combined_score >= self.threshold
                    if not post.is_relevant:
                        logger.debug("BERT опроверг пост, который keyword посчитал релевантным")
                post.relevance_score = combined_score
            else:
                # Если keyword filter не нашёл релевантность
                if score >= 0.7 and post.relevance_score >= 0:
                    # Высокий порог для ML-only решений
                    post.is_relevant = True
                    post.relevance_score = score
                    logger.debug("BERT уверен в релевантности поста")
                else:
                    if post.relevance_score >= 0:
                        post.relevance_score = combined_score

            logger.debug(f"Итоговый статус - Релевантность:{post.is_relevant} Score:({post.relevance_score:.3f})")

        relevant_count = sum(1 for p in posts if p.is_relevant)
        logger.info(f"BERT классификация: {relevant_count}/{len(posts)} постов релевантны туризму")

        return posts
    