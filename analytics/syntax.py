"""
Синтаксические связи «топоним -> слово».

Natasha (morph + syntax) строит дерево зависимостей предложения; вытаскиваем:
  - глагол, которому подчинён топоним (что делают на/с топонимом): «съездить на Ольхон»;
  - прилагательное-определение топонима: «зимний Байкал».
Лемма слова — pymorphy3.
"""
from __future__ import annotations

import pandas as pd
from natasha import Doc

from . import _nlp
from .toponyms import normalize_toponym

SYNTAX_COLS = ["post_uid", "post_id", "sentence_idx", "toponym", "word", "normal_form", "pos", "deprel"]

# отношения, по которым топоним подчинён глаголу-действию
_HEAD_RELS = {"obl", "nsubj", "obj", "nmod", "iobj"}


def _lemma(word: str) -> str:
    return _nlp.morph().parse(word)[0].normal_form


def extract_syntax_from_sentence(sentence: str) -> list[dict]:
    """Связи [{toponym, word, normal_form, pos, deprel}] для одного предложения."""
    if not sentence or not sentence.strip():
        return []

    doc = Doc(sentence)
    doc.segment(_nlp.segmenter())
    doc.tag_ner(_nlp.ner())
    doc.tag_morph(_nlp.morph_tagger())
    doc.parse_syntax(_nlp.syntax_parser())

    tokens = [t for s in doc.sents for t in s.tokens]
    by_id = {t.id: t for t in tokens}

    links: list[dict] = []
    for span in doc.spans:
        if span.type != "LOC":
            continue
        toponym = normalize_toponym(span.text)
        span_toks = [t for t in tokens if t.start >= span.start and t.stop <= span.stop]
        if not span_toks:
            continue
        ids = {t.id for t in span_toks}
        # «голова» топонимной группы — токен, чья голова вне группы
        head = next((t for t in span_toks if t.head_id not in ids), span_toks[0])

        # 1) глагол-действие — синтаксическая голова топонима
        gov = by_id.get(head.head_id)
        if gov is not None and gov.pos == "VERB" and head.rel in _HEAD_RELS:
            links.append({"toponym": toponym, "word": gov.text,
                          "normal_form": _lemma(gov.text), "pos": "VERB", "deprel": head.rel})

        # 2) прилагательные-определения топонима (amod)
        for t in tokens:
            if t.head_id in ids and t.pos == "ADJ" and t.rel == "amod":
                links.append({"toponym": toponym, "word": t.text,
                              "normal_form": _lemma(t.text), "pos": "ADJ", "deprel": "amod"})

    return links


def extract_syntax(sentences_df: pd.DataFrame) -> pd.DataFrame:
    """По DataFrame предложений возвращает связи топоним -> слово/глагол."""
    rows: list[dict] = []
    for rec in sentences_df.to_dict("records"):
        for link in extract_syntax_from_sentence(rec.get("sentence", "")):
            rows.append({
                "post_uid": rec.get("post_uid"),
                "post_id": rec.get("post_id"),
                "sentence_idx": rec.get("sentence_idx"),
                **link,
            })
    return pd.DataFrame(rows, columns=SYNTAX_COLS)
