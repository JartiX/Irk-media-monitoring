"""
Слой аналитики топонимов.

Пайплайн: загрузка постов -> сегментация -> извлечение топонимов -> геокодинг ->
синтаксис (топоним->глагол) -> тематика -> тональность -> выгрузка в Supabase/CSV.
"""

from .loader import load_vk_xlsx, load_gazetteer, load_supabase_posts
from .preprocess import clean_text, split_sentences, sentences_frame
from .toponyms import extract_toponyms, extract_from_sentence, normalize_toponym
from .geocode import geocode_toponym, geocode_frame
from .syntax import extract_syntax, extract_syntax_from_sentence
from .topics import classify, classify_series, CATEGORIES
from .sentiment import sentence_sentiment, add_sentiment, tonality_by_toponym

__all__ = [
    "load_vk_xlsx",
    "load_gazetteer",
    "load_supabase_posts",
    "clean_text",
    "split_sentences",
    "sentences_frame",
    "extract_toponyms",
    "extract_from_sentence",
    "normalize_toponym",
    "geocode_toponym",
    "geocode_frame",
    "extract_syntax",
    "extract_syntax_from_sentence",
    "classify",
    "classify_series",
    "CATEGORIES",
    "sentence_sentiment",
    "add_sentiment",
    "tonality_by_toponym",
]
