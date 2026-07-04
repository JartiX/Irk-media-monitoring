"""
Извлечение топонимов.

Источники топонимов:
  1) Natasha NER (тип LOC) — основной;
  2) словарь региональных топонимов проекта `patterns/geo.py` — добивает то, что NER пропустил.
Нормализация формы — pymorphy3 (лемма каждого слова, lower): «Байкале» -> «байкал».

Модели Natasha/pymorphy3 грузятся лениво (один раз на процесс).
"""
from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd
from natasha import Doc

from patterns.geo import GEO_REGEX
from . import _nlp

# слова из спана топонима (кириллица, с дефисом: Усть-Баргузин)
_WORD = re.compile(r"[А-Яа-яЁё][А-Яа-яЁё\-]*")

MENTION_COLS = [
    "post_uid", "post_id", "source", "group_id", "published_at",
    "sentence_idx", "sentence", "toponym", "word", "method",
]


def _models():
    """Сегментатор, NER, pymorphy3 — из общего модуля _nlp (грузятся один раз)."""
    return _nlp.segmenter(), _nlp.ner(), _nlp.morph()


@lru_cache(maxsize=200_000)
def normalize_toponym(span: str) -> str:
    """Нормализует спан топонима: лемма каждого слова, lower, через пробел."""
    _, _, morph = _models()
    tokens = _WORD.findall(span)
    if not tokens:
        return span.strip().lower()
    return " ".join(morph.parse(t)[0].normal_form for t in tokens).lower()


def extract_from_sentence(sentence: str) -> list[dict]:
    """Возвращает уникальные топонимы предложения: [{toponym, word, method}]."""
    seg, ner, _ = _models()
    if not sentence or not sentence.strip():
        return []

    doc = Doc(sentence)
    doc.segment(seg)
    doc.tag_ner(ner)

    found: list[dict] = []
    seen: set[str] = set()

    # 1) NER LOC
    for span in doc.spans:
        if span.type != "LOC":
            continue
        norm = normalize_toponym(span.text)
        if norm and norm not in seen:
            seen.add(norm)
            found.append({"toponym": norm, "word": span.text, "method": "ner"})

    # 2) словарь региональных топонимов (то, что NER пропустил)
    for m in GEO_REGEX.finditer(sentence):
        word = m.group(0)
        norm = normalize_toponym(word)
        if norm and norm not in seen:
            seen.add(norm)
            found.append({"toponym": norm, "word": word, "method": "dict"})

    return found


def extract_toponyms(sentences_df: pd.DataFrame) -> pd.DataFrame:
    """По DataFrame предложений возвращает DataFrame упоминаний топонимов."""
    rows: list[dict] = []
    for rec in sentences_df.to_dict("records"):
        for f in extract_from_sentence(rec.get("sentence", "")):
            rows.append({
                "post_uid": rec.get("post_uid"),
                "post_id": rec.get("post_id"),
                "source": rec.get("source"),
                "group_id": rec.get("group_id"),
                "published_at": rec.get("published_at"),
                "sentence_idx": rec.get("sentence_idx"),
                "sentence": rec.get("sentence"),
                **f,
            })
    return pd.DataFrame(rows, columns=MENTION_COLS)
