"""
Очистка текста и сегментация на предложения.

Единица анализа дальше — предложение (чтобы привязывать топоним к тональности и действиям).
Сегментация — razdel.sentenize (входит в стек natasha, не требует загрузки моделей).
"""
from __future__ import annotations

import re
from typing import Iterator

import pandas as pd
from razdel import sentenize

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
# эмодзи / пиктограммы / символы
_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "️‍"
    "]+",
    flags=re.UNICODE,
)
_MULTI_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Удаляет ссылки, e-mail, эмодзи, схлопывает пробелы."""
    if not isinstance(text, str):
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""
        text = str(text)
    t = _URL.sub(" ", text)
    t = _EMAIL.sub(" ", t)
    t = _EMOJI.sub(" ", t)
    t = t.replace("\xa0", " ").replace("​", " ")
    t = _MULTI_WS.sub(" ", t).strip()
    return t


def split_sentences(text: str) -> list[str]:
    """Очищает текст и разбивает на предложения."""
    text = clean_text(text)
    if not text:
        return []
    return [s.text.strip() for s in sentenize(text) if s.text.strip()]


def iter_sentences(
    df: pd.DataFrame, text_col: str = "text", id_col: str = "post_uid"
) -> Iterator[dict]:
    """Генерирует записи предложений с метаданными поста."""
    for _, row in df.iterrows():
        for idx, sent in enumerate(split_sentences(row[text_col])):
            yield {
                "post_uid": row.get(id_col),
                "post_id": row.get("post_id"),
                "source": row.get("source"),
                "group_id": row.get("group_id"),
                "published_at": row.get("published_at"),
                "sentence_idx": idx,
                "sentence": sent,
            }


def sentences_frame(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """DataFrame предложений: post_uid, post_id, source, group_id, published_at, sentence_idx, sentence."""
    rows = list(iter_sentences(df, **kwargs))
    cols = ["post_uid", "post_id", "source", "group_id", "published_at", "sentence_idx", "sentence"]
    return pd.DataFrame(rows, columns=cols)
