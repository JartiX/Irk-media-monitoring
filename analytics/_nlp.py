"""
Ленивые синглтоны NLP-моделей Natasha + pymorphy3.

Общие для извлечения топонимов и синтаксиса, чтобы NewsEmbedding и прочие модели
грузились один раз на процесс (а не дублировались).
"""
from __future__ import annotations

import pymorphy3
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
)

_state: dict = {}


def segmenter() -> Segmenter:
    if "seg" not in _state:
        _state["seg"] = Segmenter()
    return _state["seg"]


def _emb() -> NewsEmbedding:
    if "emb" not in _state:
        _state["emb"] = NewsEmbedding()
    return _state["emb"]


def ner() -> NewsNERTagger:
    if "ner" not in _state:
        _state["ner"] = NewsNERTagger(_emb())
    return _state["ner"]


def morph_tagger() -> NewsMorphTagger:
    if "mt" not in _state:
        _state["mt"] = NewsMorphTagger(_emb())
    return _state["mt"]


def syntax_parser() -> NewsSyntaxParser:
    if "sp" not in _state:
        _state["sp"] = NewsSyntaxParser(_emb())
    return _state["sp"]


def morph() -> "pymorphy3.MorphAnalyzer":
    if "pm" not in _state:
        _state["pm"] = pymorphy3.MorphAnalyzer()
    return _state["pm"]
