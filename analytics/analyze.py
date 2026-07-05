"""
Разбор предложения за один проход Natasha: топонимы + тональность/тема каждого топонима
+ синтаксические связи (топоним -> глагол/прилагательное).

Тональность и тему относим к ближайшему топониму: в предложениях с несколькими местами
нельзя присваивать балл всего предложения всем одинаково.
"""
from __future__ import annotations

import re

import pandas as pd
from natasha import Doc

from . import _nlp
from .toponyms import normalize_toponym
from .sentiment import _POS_RX, _NEG_RX
from .topics import _COMPILED as _TOPIC_RX
from .canonical_geo import TOPONYM_STOP
from .osm_gazetteer import osm_names
from patterns.geo import GEO_REGEX

# ML-тональность для предложений с одним топонимом (выкл. по умолчанию)
USE_ML_SENTIMENT = False

_HEAD_RELS = {"obl", "nsubj", "obj", "nmod", "iobj"}
# отношения, по которым берём существительное-вершину топонима («берег Байкала», «остров Ольхон»)
_NOUN_RELS = _HEAD_RELS | {"appos", "flat", "conj"}
# слишком общие существительные — не характеризуют место
NOUN_STOP = {"год", "день", "время", "раз", "человек", "место", "часть", "число", "количество",
             "километр", "метр", "рубль", "процент", "случай", "вопрос", "работа", "данные",
             "информация", "конец", "начало", "неделя", "месяц", "сутки", "название"}
# частицы/интенсификаторы, сквозь которые «не» всё ещё действует на оценочное слово
_PARTICLES = {"очень", "так", "совсем", "особо", "уж", "же", "то", "-то", "настолько", "столь", "слишком"}

MENTION_COLS = ["post_uid", "post_id", "source", "group_id", "published_at",
                "sentence_idx", "sentence", "toponym", "word",
                "sentence_sentiment", "topic_category"]
SYNTAX_COLS = ["post_uid", "post_id", "sentence_idx", "toponym", "word", "normal_form", "pos",
               "deprel", "sentence_sentiment"]


def _lemma(word: str) -> str:
    return _nlp.morph().parse(word)[0].normal_form


_VERBAL = {"INFN", "VERB", "GRND", "PRTF", "PRTS"}


def _is_nominal(tok) -> bool:
    """Токен — именная часть топонима? PROPN всегда; NOUN/ADJ — если pymorphy не считает их глаголом."""
    if tok.pos == "PROPN":
        return True
    if tok.pos in ("NOUN", "ADJ"):
        return _nlp.morph().parse(tok.text)[0].tag.POS not in _VERBAL
    return False


def _word_bounded(s: str, a: int, b: int) -> bool:
    """Совпадение [a:b) не часть более длинного слова (нет буквы по краям)."""
    before = s[a - 1] if a > 0 else " "
    after = s[b] if b < len(s) else " "
    return not (before.isalpha() or after.isalpha())


_CAMEL = re.compile(r"[а-яё][А-ЯЁ]")
# UPOS, которые точно не топонимы (наречия/частицы/предлоги/местоимения/…)
_REJECT_POS = {"ADV", "ADP", "CCONJ", "SCONJ", "PART", "PRON", "INTJ", "DET", "NUM", "AUX"}
# слова-маркеры улиц
_STREET_WORDS = {"ул", "улица", "улице", "проспект", "просп", "переулок", "пер",
                 "набережная", "набережной", "площадь", "бульвар", "мкр", "микрорайон"}
# служебные глаголы — не относим к «что делать на топониме»
VERB_STOP = {
    "быть", "стать", "становиться", "являться", "мочь", "хотеть", "иметь", "дать", "давать",
    "сделать", "делать", "существовать", "оказаться", "оказываться", "начать", "начинать",
    "продолжать", "считать", "сообщать", "сообщить", "рассказать", "рассказывать", "сказать",
    "говорить", "происходить", "случиться", "получить", "получать", "находиться", "знать",
}


def _reject_toponym(text: str, norm: str, head) -> bool:
    """Фильтр ложных топонимов: хэштеги, нарицательные, наречия, имена людей."""
    if _CAMEL.search(text):              # camelCase из хэштега
        return True
    if norm in TOPONYM_STOP:             # нарицательные/стоп-слова
        return True
    if head is not None and " " not in norm and head.pos in _REJECT_POS:
        return True
    if " " not in norm:                  # проверка части речи через pymorphy
        pos = _nlp.morph().parse(text)[0].tag.POS
        if pos in {"ADVB", "PRCL", "CONJ", "PREP", "NPRO", "INTJ"}:
            return True
    return False


def analyze_sentence(sentence: str):
    """Возвращает (mentions, links) для одного предложения.

    mentions: [{toponym, word, sentence_sentiment, topic_category}] — тон./тема привязаны к топониму.
    links:    [{toponym, word, normal_form, pos, deprel}] — синтаксис топоним -> слово.
    """
    if not sentence or not sentence.strip():
        return [], []

    doc = Doc(sentence)
    doc.segment(_nlp.segmenter())
    doc.tag_ner(_nlp.ner())
    doc.tag_morph(_nlp.morph_tagger())
    doc.parse_syntax(_nlp.syntax_parser())

    tokens = [t for s in doc.sents for t in s.tokens]
    by_id = {t.id: t for t in tokens}

    # 1) топонимы (LOC), дедуп по норм. форме
    topos = []
    seen = set()
    for span in doc.spans:
        if span.type != "LOC":
            continue
        span_toks = [t for t in tokens if t.start >= span.start and t.stop <= span.stop]
        if not span_toks:
            continue
        # берём непрерывный блок именных токенов (разрыв на не-именном токене или пунктуации)
        runs, cur, prev = [], [], None
        for t in span_toks:
            nominal = _is_nominal(t)
            punct_gap = prev is not None and sentence[prev.stop:t.start].strip() != ""
            if nominal and not punct_gap:
                cur.append(t)
            elif nominal and punct_gap:
                if cur:
                    runs.append(cur)
                cur = [t]
            else:
                if cur:
                    runs.append(cur)
                cur = []
            prev = t
        if cur:
            runs.append(cur)
        core = max(runs, key=len) if runs else span_toks
        word = " ".join(t.text for t in core)
        norm = normalize_toponym(word)
        if not norm or norm in seen:
            continue
        ids = {t.id for t in core}
        head = next((t for t in core if t.head_id not in ids), core[0])
        if _reject_toponym(word, norm, head):
            continue
        seen.add(norm)
        topos.append({"norm": norm, "word": word, "start": core[0].start, "stop": core[-1].stop,
                      "ids": ids, "head": head})

    # словарь региональных топонимов (что пропустил NER)
    for mt in GEO_REGEX.finditer(sentence):
        if not _word_bounded(sentence, mt.start(), mt.end()):
            continue  # не матчим подстроки внутри слов
        norm = normalize_toponym(mt.group(0))
        if not norm or norm in seen:
            continue
        span_toks = [t for t in tokens if t.start <= mt.start() < t.stop]
        ids = {t.id for t in span_toks}
        head = next((t for t in span_toks if t.head_id not in ids), span_toks[0]) if span_toks else None
        if _reject_toponym(mt.group(0), norm, head):
            continue
        seen.add(norm)
        topos.append({"norm": norm, "word": mt.group(0), "start": mt.start(), "stop": mt.end(),
                      "ids": ids, "head": head})

    # имена собственные из OSM-газеттира, которые пропустил NER
    gaz = osm_names()
    if gaz:
        for idx, tok in enumerate(tokens):
            if tok.pos != "PROPN" or not tok.text[:1].isupper():
                continue
            j = idx - 1                    # предыдущий непунктуационный токен
            while j >= 0 and not tokens[j].text[:1].isalpha():
                j -= 1
            prev = tokens[j].text.lower().rstrip(".") if j >= 0 else ""
            if prev in _STREET_WORDS:      # пропускаем улицы
                continue
            norm = normalize_toponym(tok.text)
            if norm in gaz and norm not in seen and not _reject_toponym(tok.text, norm, tok):
                seen.add(norm)
                topos.append({"norm": norm, "word": tok.text, "start": tok.start, "stop": tok.stop,
                              "ids": {tok.id}, "head": tok})

    if not topos:
        return [], []

    def _dist(t, pos: int) -> int:
        if t["start"] <= pos <= t["stop"]:
            return 0
        return min(abs(t["start"] - pos), abs(t["stop"] - pos))

    def nearest(pos: int) -> str:
        return min(topos, key=lambda t: _dist(t, pos))["norm"]

    # 2) привязка тональности и темы к ближайшему топониму
    sent = {t["norm"]: 0.0 for t in topos}
    topic_counts = {t["norm"]: {} for t in topos}
    lemmas = [_nlp.morph().parse(t.text)[0].normal_form for t in tokens]

    for i, tok in enumerate(tokens):
        probe = tok.text + " " + lemmas[i]  # матчим и по форме, и по лемме
        pol = 1 if _POS_RX.search(probe) else (-1 if _NEG_RX.search(probe) else 0)
        if pol != 0:  # отрицание «не» перед словом инвертирует оценку
            for j in range(i - 1, max(-1, i - 4), -1):
                w = tokens[j].text.lower()
                if w == "не":
                    pol = -pol
                    break
                if w not in _PARTICLES:
                    break
        if pol != 0:
            sent[nearest(tok.start)] += pol
        for cat, rx in _TOPIC_RX.items():
            if rx.search(probe):
                near = nearest(tok.start)
                topic_counts[near][cat] = topic_counts[near].get(cat, 0) + 1
                break

    # 3) синтаксис: топоним -> глагол (голова) / прилагательное (amod)
    links = []
    for t in topos:
        if t["head"] is None:
            continue
        gov = by_id.get(t["head"].head_id)
        if gov is not None and gov.pos == "VERB" and t["head"].rel in _HEAD_RELS:
            lem = _lemma(gov.text)
            if lem not in VERB_STOP:      # без служебных глаголов
                links.append({"toponym": t["norm"], "word": gov.text,
                              "normal_form": lem, "pos": "VERB", "deprel": t["head"].rel})
        elif gov is not None and gov.pos == "NOUN" and t["head"].rel in _NOUN_RELS:
            lem = _lemma(gov.text)
            if lem not in NOUN_STOP and len(lem) > 2:   # существительное-вершина топонима
                links.append({"toponym": t["norm"], "word": gov.text,
                              "normal_form": lem, "pos": "NOUN", "deprel": t["head"].rel})
        for tok in tokens:
            if tok.head_id in t["ids"] and tok.pos == "ADJ" and tok.rel == "amod":
                links.append({"toponym": t["norm"], "word": tok.text,
                              "normal_form": _lemma(tok.text), "pos": "ADJ", "deprel": "amod"})

    # ML-тональность для предложений с одним топонимом; приводим к шкале лексикона (±1/0)
    if USE_ML_SENTIMENT and len(topos) == 1:
        from .sentiment_ml import ml_sentiment
        s = ml_sentiment(sentence)
        if s is not None:
            sent[topos[0]["norm"]] = 1.0 if s >= 0.5 else (-1.0 if s <= -0.5 else 0.0)

    # тональность предложения для каждой связи
    for lk in links:
        lk["sentence_sentiment"] = sent.get(lk["toponym"], 0.0)

    mentions = []
    for t in topos:
        tc = topic_counts[t["norm"]]
        best = max(tc, key=tc.get) if tc else ""
        mentions.append({"toponym": t["norm"], "word": t["word"],
                         "sentence_sentiment": sent[t["norm"]], "topic_category": best})
    return mentions, links


def analyze_frame(sentences_df: pd.DataFrame):
    """По DataFrame предложений -> (mentions_df, syntax_df) за один проход на предложение."""
    m_rows, s_rows = [], []
    meta_keys = ["post_uid", "post_id", "source", "group_id", "published_at", "sentence_idx", "sentence"]
    for rec in sentences_df.to_dict("records"):
        mlist, slinks = analyze_sentence(rec.get("sentence", ""))
        meta = {k: rec.get(k) for k in meta_keys}
        for m in mlist:
            m_rows.append({**meta, **m})
        for s in slinks:
            s_rows.append({"post_uid": rec.get("post_uid"), "post_id": rec.get("post_id"),
                           "sentence_idx": rec.get("sentence_idx"), **s})
    return pd.DataFrame(m_rows, columns=MENTION_COLS), pd.DataFrame(s_rows, columns=SYNTAX_COLS)
