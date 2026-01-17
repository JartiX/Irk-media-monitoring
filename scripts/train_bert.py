#!/usr/bin/env python
"""
Скрипт обучения BERT классификатора для туристического контента.

Использование:
    python scripts/train_bert.py [--epochs N] [--batch-size N] [--lr FLOAT]

Примеры:
    # Обучение с параметрами по умолчанию
    .venv\Scripts\python.exe scripts/train_bert.py

    # Обучение с кастомными параметрами
    .venv\Scripts\python.exe scripts/train_bert.py --epochs 5 --batch-size 8 --lr 1e-5
"""
import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

# Добавляем корень проекта в путь для импортов
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

# Настройка логирования
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)

# Создаем директорию для логов если не существует
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)

logger.add(
    str(logs_dir / f"bert_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Обучение BERT классификатора туристического контента"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Количество эпох обучения (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Размер батча (default: 16)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Максимальная длина токенизации (default: 256)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="JartiX/bert_tourism_classifier",
        help="Путь для сохранения модели (default: JartiX/bert_tourism_classifier)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Принудительное переобучение даже если модель существует"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Обучение BERT классификатора туристического контента")
    logger.info("=" * 60)
    logger.info(f"Параметры:")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Learning rate: {args.lr}")
    logger.info(f"  - Max length: {args.max_length}")
    logger.info(f"  - Model path: {args.model_path}")

    # Импортируем после настройки логирования
    try:
        from filters.bert_classifier import BertClassifier
        import ml
    except ImportError as e:
        logger.error(f"Ошибка импорта: {e}")
        logger.error("Убедитесь что установлены зависимости BERT: pip install -r \"requirements_bert.txt\"")
        sys.exit(1)

    # Загружаем данные
    logger.info("")
    logger.info("Загрузка обучающих данных...")
    positive = ml.POSITIVE_ML_TRAIN
    negative = ml.NEGATIVE_ML_TRAIN

    texts = positive + negative
    labels = [1] * len(positive) + [0] * len(negative)

    logger.info(f"Всего примеров: {len(texts)}")
    logger.info(f"  - Позитивных (туризм): {len(positive)}")
    logger.info(f"  - Негативных (не туризм): {len(negative)}")
    logger.info(f"  - Баланс классов: {len(positive)/len(texts)*100:.1f}% / {len(negative)/len(texts)*100:.1f}%")

    # Создаем и обучаем классификатор
    logger.info("")
    classifier = BertClassifier(model_path=args.model_path)

    # Проверка существующей модели
    if classifier.is_trained and not args.force:
        logger.info(f"Модель {args.model_path} уже существует на HF. Используйте --force для переобучения.")
        response = input("Переобучить модель? [y/N]: ").strip().lower()
        if response != "y":
            logger.info("Обучение отменено")
            return
        
    start_time = datetime.now()

    metrics = classifier.train(
        texts=texts,
        labels=labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )

    elapsed = (datetime.now() - start_time).total_seconds()

    # Результаты
    logger.info("")
    logger.info("=" * 60)
    logger.info("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
    logger.info("=" * 60)
    logger.info(f"Время обучения: {elapsed:.1f} секунд ({elapsed/60:.1f} минут)")

    if "error" not in metrics:
        logger.info(f"Train Loss: {metrics['train_loss']:.4f}")
        logger.info(f"Validation Accuracy: {metrics['eval_accuracy']:.4f}")
        logger.info(f"Validation F1: {metrics['eval_f1']:.4f}")
        logger.info(f"Train samples: {metrics['train_samples']}")
        logger.info(f"Validation samples: {metrics['val_samples']}")
        logger.info(f"Модель сохранена в: {args.model_path}")
    else:
        logger.error(f"Ошибка обучения: {metrics['error']}")
        sys.exit(1)

    # Тестовые предсказания
    logger.info("")
    logger.info("=" * 60)
    logger.info("ТЕСТОВЫЕ ПРЕДСКАЗАНИЯ")
    logger.info("=" * 60)

    test_texts = [
        # Позитивные примеры (должны быть релевантны)
        ("Экскурсия на Байкал с посещением острова Ольхон", True),
        ("Новый отель открылся в Листвянке с видом на озеро", True),
        ("Треккинг по Тункинской долине с гидом", True),
        ("Фестиваль льда на Байкале собрал тысячи туристов", True),
        ("Глэмпинг на берегу Байкала - комфортный отдых", True),

        # Негативные примеры (не должны быть релевантны)
        ("Авария на теплотрассе оставила без отопления жителей", False),
        ("Выборы губернатора назначены на сентябрь", False),
        ("Цены на бензин выросли на 5%", False),
        ("Прогноз погоды: сильные морозы до -30", False),
        ("Задержан подозреваемый в краже", False),
    ]

    correct = 0
    for text, expected in test_texts:
        is_rel, score = classifier.predict(text)
        status = "+" if is_rel else "-"
        match = "OK" if is_rel == expected else "FAIL"
        if is_rel == expected:
            correct += 1
        logger.info(f"  [{status}] {score:.3f} [{match}] | {text[:60]}...")

    logger.info("")
    logger.info(f"Точность на тестовых примерах: {correct}/{len(test_texts)} ({correct/len(test_texts)*100:.0f}%)")
    logger.info("")
    logger.info("Обучение завершено успешно!")


if __name__ == "__main__":
    main()
