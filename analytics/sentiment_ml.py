"""
ML-тональность на предобученной модели.

Модель cointegrated/rubert-tiny-sentiment-balanced (3 класса negative/neutral/positive).
ml_sentiment(text) -> скаляр P(pos) - P(neg) в [-1, 1] (или None, если модель/transformers
недоступны — тогда используется лексикон).

Ленивая загрузка; включается флагом analyze.USE_ML_SENTIMENT (по умолчанию выкл. из-за скорости).
"""
from __future__ import annotations

MODEL = "cointegrated/rubert-tiny-sentiment-balanced"

_state = None  # (tokenizer, model, torch) | False (не удалось)


def _pipe():
    global _state
    if _state is None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tok = AutoTokenizer.from_pretrained(MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL)
            model.eval()
            _state = (tok, model, torch)
        except Exception:
            _state = False
    return _state


def available() -> bool:
    return _pipe() is not False


def ml_sentiment(text: str) -> float | None:
    """Тональность текста в [-1, 1] (P_pos − P_neg) или None, если модель недоступна."""
    p = _pipe()
    if p is False or not text:
        return None
    tok, model, torch = p
    with torch.no_grad():
        inputs = tok(str(text)[:512], return_tensors="pt", truncation=True, max_length=256)
        probs = torch.softmax(model(**inputs).logits[0], dim=-1).tolist()
    id2label = model.config.id2label
    score = 0.0
    for i, pr in enumerate(probs):
        lab = str(id2label.get(i, "")).lower()
        if "pos" in lab:
            score += pr
        elif "neg" in lab:
            score -= pr
    return float(score)
