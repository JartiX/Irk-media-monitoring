"""
Данные для дашборда: Supabase (прод) с CSV-fallback (dev).
Без Streamlit-зависимостей — функции тестируемы отдельно.
"""
from __future__ import annotations

import pandas as pd

CSV_DIR = "data/analytics"

# bbox региона (lat_min, lat_max, lon_min, lon_max) — на карту попадают только точки внутри региона
REGION_BBOX = (51.0, 64.0, 96.0, 119.0)

_M_COLS = ["post_id", "toponym_name", "source_type", "published_at", "sentence_sentiment", "topic_category"]
# слова синтаксиса для облака слов (глаголы/прилагательные с тональностью предложения)
_S_COLS = ["post_id", "toponym_name", "word", "normal_form", "pos", "sentence_sentiment"]


def _all(c, table, cols):
    """Постранично читает все строки (PostgREST отдаёт max ~1000 за запрос)."""
    rows, off, page = [], 0, 1000
    while True:
        b = c.table(table).select(cols).range(off, off + page - 1).execute().data
        if not b:
            break
        rows.extend(b)
        if len(b) < page:
            break
        off += page
    return rows


def _from_supabase():
    from database.supabase_client import SupabaseClient
    c = SupabaseClient().client
    mentions = pd.DataFrame(_all(c, "toponym_mentions", ",".join(_M_COLS)))
    toponyms = pd.DataFrame(_all(c, "toponyms", "name,display_name,lat,lon"))
    syntax = pd.DataFrame(_all(c, "toponym_syntax", ",".join(_S_COLS)))
    return mentions, toponyms, syntax


def _from_csv():
    m = pd.read_csv(f"{CSV_DIR}/toponym_mentions.csv").rename(
        columns={"place": "toponym_name", "source": "source_type"})
    t = pd.read_csv(f"{CSV_DIR}/toponyms.csv")
    s = pd.read_csv(f"{CSV_DIR}/toponym_syntax.csv").rename(columns={"place": "toponym_name"})
    return m, t, s


def load_analytics():
    """Возвращает (mentions, toponyms, syntax, source_label)."""
    empty = pd.DataFrame()
    try:
        m, t, s = _from_supabase()
        if m is None or m.empty:
            raise ValueError("пустые таблицы Supabase")
        src = "Supabase"
    except Exception:
        try:
            m, t, s = _from_csv()
            src = "локальный CSV"
        except Exception:
            return empty, empty, empty, "нет данных"

    # нормализация полей
    if "published_at" in m:
        m["published_at"] = pd.to_datetime(m["published_at"], errors="coerce", utc=True)
    m["sentence_sentiment"] = pd.to_numeric(m.get("sentence_sentiment"), errors="coerce").fillna(0.0)
    m["topic_category"] = (m["topic_category"].fillna("") if "topic_category" in m else "")
    m["source_type"] = (m["source_type"].fillna("unknown") if "source_type" in m else "unknown")
    if s is not None and not s.empty and "sentence_sentiment" in s:
        s["sentence_sentiment"] = pd.to_numeric(s["sentence_sentiment"], errors="coerce").fillna(0.0)
    return m, t, s, src


_KIND_BY_POS = {"VERB": "глагол", "ADJ": "прилагательное"}


def word_stats(mentions: pd.DataFrame, syntax: pd.DataFrame, post_ids=None) -> pd.DataFrame:
    """Единая частотно-тональная таблица слов для облака: word, kind, freq, sentiment.

    Топонимы — из упоминаний (тональность мест), глаголы/прилагательные — из синтаксиса
    (тональность предложений, где встречается слово). post_ids ограничивает набор публикаций
    (для согласования с фильтрами дашборда).
    """
    parts = []
    m = mentions
    s = syntax
    if post_ids is not None:
        if m is not None and "post_id" in m:
            m = m[m["post_id"].isin(post_ids)]
        if s is not None and "post_id" in s:
            s = s[s["post_id"].isin(post_ids)]

    if m is not None and not m.empty and "toponym_name" in m:
        g = (m.groupby("toponym_name")
             .agg(freq=("toponym_name", "size"), sentiment=("sentence_sentiment", "mean"))
             .reset_index().rename(columns={"toponym_name": "word"}))
        g["kind"] = "топоним"
        parts.append(g)

    if s is not None and not s.empty and "pos" in s and "normal_form" in s:
        sv = s.copy()
        sv["sentiment"] = pd.to_numeric(sv.get("sentence_sentiment"), errors="coerce").fillna(0.0)
        for pos, kind in _KIND_BY_POS.items():
            d = (sv[sv["pos"] == pos].dropna(subset=["normal_form"])
                 .groupby("normal_form")
                 .agg(freq=("normal_form", "size"), sentiment=("sentiment", "mean"))
                 .reset_index().rename(columns={"normal_form": "word"}))
            if not d.empty:
                d["kind"] = kind
                parts.append(d)

    if not parts:
        return pd.DataFrame(columns=["word", "kind", "freq", "sentiment"])
    out = pd.concat(parts, ignore_index=True)
    out = out[out["word"].astype(str).str.len() > 1]  # отсекаем односимвольный мусор
    return out.reset_index(drop=True)


def aggregate(mentions: pd.DataFrame, toponyms: pd.DataFrame, sources=None,
              date_from=None, date_to=None, topics=None):
    """Фильтрует упоминания и агрегирует по топониму: n, sentiment, pos, neg, lat, lon."""
    cols = ["toponym_name", "n", "sentiment", "pos", "neg", "lat", "lon"]
    m = mentions.copy()
    if sources:
        m = m[m["source_type"].isin(sources)]
    if topics:
        m = m[m["topic_category"].isin(topics)]
    if date_from is not None:
        m = m[m["published_at"] >= date_from]
    if date_to is not None:
        m = m[m["published_at"] <= date_to]
    if m.empty:
        return m, pd.DataFrame(columns=cols)

    agg = m.groupby("toponym_name").agg(
        n=("toponym_name", "size"),
        sentiment=("sentence_sentiment", "sum"),
        pos=("sentence_sentiment", lambda s: float(s[s > 0].sum())),
        neg=("sentence_sentiment", lambda s: float(s[s < 0].sum())),
    ).reset_index()

    if toponyms is not None and not toponyms.empty and "name" in toponyms:
        coords = toponyms.rename(columns={"name": "toponym_name"})[["toponym_name", "lat", "lon"]]
        agg = agg.merge(coords, on="toponym_name", how="left")
        # на карту — только точки региона; координаты вне bbox скрываем, топоним остаётся в частотах
        la0, la1, lo0, lo1 = REGION_BBOX
        lat = pd.to_numeric(agg["lat"], errors="coerce")
        lon = pd.to_numeric(agg["lon"], errors="coerce")
        out = ~(lat.between(la0, la1) & lon.between(lo0, lo1))
        agg.loc[out, ["lat", "lon"]] = None
    else:
        agg["lat"] = None
        agg["lon"] = None
    return m, agg.sort_values("n", ascending=False).reset_index(drop=True)
