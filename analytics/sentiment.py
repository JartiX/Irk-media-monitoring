"""
Тональность по топониму (лексиконный подход).

Оцениваем тональность предложения, в котором встречается топоним:
score = (число позитивных слов) - (число негативных), с обработкой отрицания «не <позитив>».
Затем агрегируем по топониму (positiv / negative / sum_ton).
"""
from __future__ import annotations

import re

import pandas as pd

_POS = [
    "прекрасн", "красив", "красот", "отличн", "чудесн", "шикарн", "великолепн", "восхитительн",
    "потрясающ", "потряс", "удивительн", "лучш", "чист", "уютн", "душевн", "рекоменд", "совету",
    "понрав", "влюбил", "влюбл", "райск", "сказочн", "сказк", "живописн", "незабыва", "волшебн",
    "кайф", "классн", r"\bкрут", "супер", "обожа", "нрав", "впечатл", "комфорт", "приятн",
    "восторг", r"\bрад\b", "спокойн", "гостеприим", "восхищ", "благодар", "замечательн",
    "обалденн", "офигенн", "бесподобн", "наслажд", "довольн", "счастлив", "любим",
    "шедевр", "идеальн", "завораж", "умиротвор", "добродушн", "великолеп", "тёпл",
]
_NEG = [
    "ужасн", "отвратительн", "отврат", "грязн", "грязищ", r"\bплох", "паршив", "разочаров",
    "разочарование", r"\bмусор", "помойк", "свалк", "опасн", r"груб(?!о\s+говоря)", "хамств",
    "хамл", r"нагл(?!ядн)", "обман", "мошенн", r"\bужас", "кошмар", "отстой", "недовол",
    r"\bжаль", "убог", r"разруш(?!ить\s+(?:стереотип|миф|шаблон))", "разрух", "запущенн",
    "загажен", "антисанитар", "вон[ья]",
    "проблем", "ругал", "втридорог", "переплат", r"\bдорого\b", r"дорог(ущ|оват|овизн)",
    r"\bтолп", "очеред", "шумн", "тесн", "неудобн",
]

_POS_RX = re.compile("|".join(_POS), re.IGNORECASE)
_NEG_RX = re.compile("|".join(_NEG), re.IGNORECASE)
# отрицание перед позитивом: «не понравилось», «не очень красиво» -> негатив
_NEG_POS_RX = re.compile(r"\bне\s+(?:очень\s+|так\s+|особо\s+)?(?:" + "|".join(_POS) + ")", re.IGNORECASE)


def sentence_sentiment(text: str) -> float:
    """Тональность предложения: >0 позитив, <0 негатив, 0 нейтрально."""
    if not text:
        return 0.0
    negated = len(_NEG_POS_RX.findall(text))
    pos = len(_POS_RX.findall(text)) - negated
    neg = len(_NEG_RX.findall(text)) + negated
    return float(max(pos, 0) - neg)


def add_sentiment(mentions_df: pd.DataFrame, text_col: str = "sentence") -> pd.DataFrame:
    """Добавляет колонку sentence_sentiment к упоминаниям."""
    df = mentions_df.copy()
    df["sentence_sentiment"] = df[text_col].map(sentence_sentiment)
    return df


def tonality_by_toponym(mentions_df: pd.DataFrame, toponym_col: str = "toponym",
                        sent_col: str = "sentence_sentiment") -> pd.DataFrame:
    """Агрегат тональности по топониму: positiv / negative / sum_ton / n (аналог Toponims_ton)."""
    g = mentions_df.groupby(toponym_col)[sent_col]
    out = pd.DataFrame({
        "positiv": g.apply(lambda s: s[s > 0].sum()),
        "negative": g.apply(lambda s: s[s < 0].sum()),
        "sum_ton": g.sum(),
        "n": g.size(),
    }).reset_index()
    return out.sort_values("sum_ton", ascending=False)
