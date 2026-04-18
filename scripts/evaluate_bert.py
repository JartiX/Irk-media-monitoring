#!/usr/bin/env python
"""
Оценка BERT-классификатора: baseline (frozen rubert-tiny2 + LogReg) vs
дообученная модель. Прогон выполняется на нескольких random_seed-ах, чтобы
отделить устойчивый прирост качества от шума разбиения train/val/test.

Для каждого seed:
  - корпус стратифицированно разбивается на train/val/test (70/15/15);
  - baseline: замороженный энкодер rubert-tiny2 + LogisticRegression на
    pooler_output train-части, предсказание на test;
  - fine-tuned: либо локальное дообучение rubert-tiny2, либо готовая
    модель из HuggingFace Hub (--finetuned-model), предсказание на test.

Выход:
    docs/assets/confusion_matrix_baseline.png   (суммарно по seed-ам)
    docs/assets/confusion_matrix_finetuned.png  (суммарно по seed-ам)
    docs/assets/metrics_per_seed.png            (bar chart с error-bars)
    docs/assets/metrics_comparison.json         (per-seed + aggregate)

Использование:
    .venv\\Scripts\\python.exe scripts/evaluate_bert.py
    .venv\\Scripts\\python.exe scripts/evaluate_bert.py --seeds 13,42,100
    .venv\\Scripts\\python.exe scripts/evaluate_bert.py --finetuned-model JartiX/bert_tourism_classifier
"""
import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)

import config

DEFAULT_SEEDS = [13, 42, 100, 2024, 7777]
TEST_SIZE = 0.15
VAL_SIZE = 0.15
BASE_MODEL = config.ML_SETTINGS.get("bert_base_model", "cointegrated/rubert-tiny2")
MAX_LENGTH = config.ML_SETTINGS.get("bert_max_length", 512)
BATCH_SIZE = config.ML_SETTINGS.get("bert_batch_size", 16)
REPORTING_MODEL_DIR = project_root / "models" / "bert_reporting"
BASELINE_MODEL_DIR = project_root / "models" / "bert_baseline"
ASSETS_DIR = project_root / "docs" / "assets"

METRIC_KEYS = ("accuracy", "precision", "recall", "f1")
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}


def load_data():
    import ml
    positive = ml.POSITIVE_ML_TRAIN
    negative = ml.NEGATIVE_ML_TRAIN
    texts = positive + negative
    labels = [1] * len(positive) + [0] * len(negative)
    logger.info(f"Загружено: {len(positive)} позитивных + {len(negative)} негативных = {len(texts)}")
    return texts, labels


def split_data(texts, labels, seed):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        texts, labels,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=labels,
    )
    val_relative = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_relative,
        random_state=seed,
        stratify=y_trainval,
    )
    logger.info(
        f"[seed={seed}] train={len(X_train)} val={len(X_val)} test={len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_pooler_embeddings(texts, tokenizer, model, device, batch_size=BATCH_SIZE):
    """Прогнать тексты через frozen BERT, вернуть pooler_output (N, hidden).
    """
    model.eval()
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.pooler_output.cpu().numpy()
            embeddings.append(pooled)
    return np.vstack(embeddings)


def evaluate_baseline(X_train, y_train, X_test, y_test, seed, save_as_hf=False):
    logger.info("")
    logger.info(f"BASELINE (seed={seed}): frozen rubert-tiny2 + LogisticRegression")

    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModel.from_pretrained(BASE_MODEL)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)

    train_emb = extract_pooler_embeddings(X_train, tokenizer, model, device)
    test_emb = extract_pooler_embeddings(X_test, tokenizer, model, device)

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(train_emb, y_train)

    y_pred = clf.predict(test_emb).tolist()

    if save_as_hf:
        save_baseline_as_hf(clf, tokenizer, seed)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return y_pred


def save_baseline_as_hf(clf, tokenizer, seed):
    """Сохранить baseline в формате AutoModelForSequenceClassification.
    """
    BASELINE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_cls = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        problem_type="single_label_classification",
    )

    coef = clf.coef_[0].astype(np.float32)
    intercept = float(clf.intercept_[0])

    hidden = coef.shape[0]
    weight = np.zeros((2, hidden), dtype=np.float32)
    bias = np.zeros(2, dtype=np.float32)
    weight[1] = coef
    bias[1] = intercept

    with torch.no_grad():
        model_cls.classifier.weight.copy_(torch.from_numpy(weight))
        model_cls.classifier.bias.copy_(torch.from_numpy(bias))

    model_cls.config.id2label = {0: "Не туризм", 1: "Туризм"}
    model_cls.config.label2id = {"Не туризм": 0, "Туризм": 1}

    model_cls.save_pretrained(str(BASELINE_MODEL_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(BASELINE_MODEL_DIR))

    meta = {
        "base_model": BASE_MODEL,
        "max_length": MAX_LENGTH,
        "threshold": 0.5,
        "random_seed": seed,
        "training": "frozen encoder + LogisticRegression on pooler_output",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    (BASELINE_MODEL_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Baseline (HF-формат, seed={seed}) сохранён: {BASELINE_MODEL_DIR}")


def evaluate_finetuned(X_train, y_train, X_val, y_val, X_test, y_test,
                       seed, pretrained_path=None):
    logger.info("")
    if pretrained_path:
        logger.info(f"FINE-TUNED (seed={seed}): готовая модель {pretrained_path}")
    else:
        logger.info(f"FINE-TUNED (seed={seed}): локальное дообучение rubert-tiny2")

    from filters.bert_classifier import BertClassifier

    set_all_seeds(seed)

    if pretrained_path:
        classifier = BertClassifier(model_path=pretrained_path)
        if not classifier.is_trained:
            raise RuntimeError(f"Не удалось загрузить модель: {pretrained_path}")
        train_metrics = {"source": "pretrained", "pretrained_path": pretrained_path}
    else:
        classifier = BertClassifier(model_path=str(REPORTING_MODEL_DIR))
        train_metrics = classifier.train(
            texts=X_train,
            labels=y_train,
            x_val=X_val,
            y_val=y_val,
            skip_push=True,
            local_save_dir=str(REPORTING_MODEL_DIR),
        )
        if "error" in train_metrics:
            raise RuntimeError(f"Ошибка обучения fine-tuned модели: {train_metrics['error']}")
        logger.info(f"Обучение завершено (seed={seed}): eval_f1={train_metrics.get('eval_f1', 0):.4f}")

    probas = classifier._predict_batch_internal(X_test, batch_size=BATCH_SIZE)
    threshold = classifier.threshold
    y_pred = [1 if p >= threshold else 0 for p in probas]

    return y_pred, train_metrics


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def plot_confusion_matrix(cm, title, output_path, subtitle=None):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Не туризм", "Туризм"])
    ax.set_yticklabels(["Не туризм", "Туризм"])
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Истина")
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=7)
    else:
        ax.set_title(title)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix сохранена: {output_path}")


def plot_metrics_bar(aggregate, seeds_count, output_path):
    """Bar chart: per-metric mean с error-bars (std) для baseline и fine-tuned."""
    x = np.arange(len(METRIC_KEYS))
    width = 0.35

    b_mean = [aggregate["baseline"][k]["mean"] for k in METRIC_KEYS]
    b_std = [aggregate["baseline"][k]["std"] for k in METRIC_KEYS]
    f_mean = [aggregate["finetuned"][k]["mean"] for k in METRIC_KEYS]
    f_std = [aggregate["finetuned"][k]["std"] for k in METRIC_KEYS]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - width / 2, b_mean, width, yerr=b_std, capsize=4,
           label="Baseline (frozen + LogReg)", color="#6B9AC4")
    ax.bar(x + width / 2, f_mean, width, yerr=f_std, capsize=4,
           label="Fine-tuned", color="#E57A44")

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[k] for k in METRIC_KEYS])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Значение метрики")
    ax.set_title(f"Метрики по {seeds_count} random seed-ам (среднее ± std)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    for xi, (mean, std) in enumerate(zip(b_mean, b_std)):
        ax.text(xi - width / 2, mean + std + 0.015, f"{mean:.3f}",
                ha="center", fontsize=8)
    for xi, (mean, std) in enumerate(zip(f_mean, f_std)):
        ax.text(xi + width / 2, mean + std + 0.015, f"{mean:.3f}",
                ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Bar chart сохранён: {output_path}")


def log_report(name, y_true, y_pred):
    logger.info(f"\nClassification report — {name}:")
    report = classification_report(
        y_true, y_pred,
        target_names=["Не туризм", "Туризм"],
        zero_division=0,
    )
    for line in report.split("\n"):
        if line.strip():
            logger.info(line)


def aggregate_metrics(per_seed):
    """Посчитать mean/std/min/max по метрикам и сумму confusion_matrix."""

    def stats_for(model_key, metric_key):
        values = [run[model_key][metric_key] for run in per_seed]
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": [float(v) for v in values],
        }

    def cm_sum(model_key):
        total = np.zeros((2, 2), dtype=int)
        for run in per_seed:
            total += np.array(run[model_key]["confusion_matrix"], dtype=int)
        return total.tolist()

    aggregate = {"baseline": {}, "finetuned": {}}
    for model_key in ("baseline", "finetuned"):
        for metric_key in METRIC_KEYS:
            aggregate[model_key][metric_key] = stats_for(model_key, metric_key)
        aggregate[model_key]["confusion_matrix_sum"] = cm_sum(model_key)

    delta = {}
    for metric_key in METRIC_KEYS:
        diffs = [
            run["finetuned"][metric_key] - run["baseline"][metric_key]
            for run in per_seed
        ]
        delta[metric_key] = {
            "mean": float(np.mean(diffs)),
            "std": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
            "min": float(np.min(diffs)),
            "max": float(np.max(diffs)),
            "values": [float(v) for v in diffs],
            "positive_seeds": int(sum(1 for d in diffs if d > 0)),
            "negative_seeds": int(sum(1 for d in diffs if d < 0)),
        }
    aggregate["delta"] = delta
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Список seed-ов через запятую. По умолчанию: 13,42,100,2024,7777.",
    )
    parser.add_argument(
        "--finetuned-model",
        default=None,
        help="HF repo id или путь к готовой fine-tuned модели. "
             "Если задано, локальное дообучение пропускается (модель одна и та же на всех seed-ах).",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("Нужен хотя бы один seed.")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTING_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Оценка BERT-классификатора на seed-ах: {seeds}")
    logger.info("=" * 60)

    texts, labels = load_data()

    per_seed = []
    split_sizes = None
    finetuned_train_info = None

    for idx, seed in enumerate(seeds):
        logger.info("")
        logger.info(f"===== Seed {seed} ({idx + 1}/{len(seeds)}) =====")

        X_train, X_val, X_test, y_train, y_val, y_test = split_data(texts, labels, seed)
        if split_sizes is None:
            split_sizes = {
                "train": len(X_train),
                "val": len(X_val),
                "test": len(X_test),
                "train_ratio": round(len(X_train) / len(texts), 4),
                "val_ratio": round(len(X_val) / len(texts), 4),
                "test_ratio": round(len(X_test) / len(texts), 4),
            }

        baseline_pred = evaluate_baseline(
            X_train, y_train, X_test, y_test,
            seed=seed,
            save_as_hf=(idx == 0),
        )
        finetuned_pred, train_metrics = evaluate_finetuned(
            X_train, y_train, X_val, y_val, X_test, y_test,
            seed=seed,
            pretrained_path=args.finetuned_model,
        )

        b_metrics = compute_metrics(y_test, baseline_pred)
        f_metrics = compute_metrics(y_test, finetuned_pred)

        log_report(f"baseline (seed={seed})", y_test, baseline_pred)
        log_report(f"fine-tuned (seed={seed})", y_test, finetuned_pred)

        per_seed.append({
            "seed": seed,
            "baseline": b_metrics,
            "finetuned": f_metrics,
        })

        if finetuned_train_info is None:
            if train_metrics.get("source") == "pretrained":
                finetuned_train_info = {
                    "source": "pretrained",
                    "pretrained_path": train_metrics.get("pretrained_path"),
                }
            else:
                finetuned_train_info = {
                    "source": "local_train",
                    "epochs": train_metrics.get("epochs"),
                    "train_loss": train_metrics.get("train_loss"),
                    "val_accuracy": train_metrics.get("eval_accuracy"),
                    "val_f1": train_metrics.get("eval_f1"),
                    "seed": seeds[0],
                }

    aggregate = aggregate_metrics(per_seed)

    total_test = split_sizes["test"] * len(seeds)
    plot_confusion_matrix(
        aggregate["baseline"]["confusion_matrix_sum"],
        "Baseline (frozen rubert-tiny2 + LogReg)",
        ASSETS_DIR / "confusion_matrix_baseline.png",
        subtitle=f"суммарно по {len(seeds)} seed-ам, всего {total_test} предсказаний",
    )
    plot_confusion_matrix(
        aggregate["finetuned"]["confusion_matrix_sum"],
        "Fine-tuned rubert-tiny2",
        ASSETS_DIR / "confusion_matrix_finetuned.png",
        subtitle=f"суммарно по {len(seeds)} seed-ам, всего {total_test} предсказаний",
    )
    plot_metrics_bar(
        aggregate,
        seeds_count=len(seeds),
        output_path=ASSETS_DIR / "metrics_per_seed.png",
    )

    baseline_top = {k: aggregate["baseline"][k]["mean"] for k in METRIC_KEYS}
    baseline_top["confusion_matrix"] = aggregate["baseline"]["confusion_matrix_sum"]
    finetuned_top = {k: aggregate["finetuned"][k]["mean"] for k in METRIC_KEYS}
    finetuned_top["confusion_matrix"] = aggregate["finetuned"]["confusion_matrix_sum"]

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seeds": seeds,
        "split": split_sizes,
        "base_model": BASE_MODEL,
        "baseline": baseline_top,
        "finetuned": finetuned_top,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "finetuned_train_info": finetuned_train_info,
    }

    metrics_path = ASSETS_DIR / "metrics_comparison.json"
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Метрики сохранены: {metrics_path}")

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"СВОДКА по {len(seeds)} seed-ам (mean +/- std)")
    logger.info("=" * 60)
    header = f"{'metric':<10} {'baseline':<20} {'fine-tuned':<20} {'delta':<20} sign"
    logger.info(header)
    for key in METRIC_KEYS:
        b = aggregate["baseline"][key]
        f = aggregate["finetuned"][key]
        d = aggregate["delta"][key]
        sign = f"+{d['positive_seeds']}/-{d['negative_seeds']}"
        logger.info(
            f"{key:<10} "
            f"{b['mean']:.4f} +/- {b['std']:.4f}   "
            f"{f['mean']:.4f} +/- {f['std']:.4f}   "
            f"{d['mean']:+.4f} +/- {d['std']:.4f}   "
            f"{sign}"
        )

    logger.info("")
    logger.info("Per-seed delta F1 (fine-tuned - baseline):")
    for run, delta_f1 in zip(per_seed, aggregate["delta"]["f1"]["values"]):
        logger.info(f"  seed={run['seed']:>5}:  {delta_f1:+.4f}")


if __name__ == "__main__":
    main()
