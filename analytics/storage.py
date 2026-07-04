"""
Запись результатов аналитики — локальный CSV и/или Supabase (REST).

- write_csv(): отладочный/дев-режим, читается дашбордом локально (всегда работает).
- write_supabase(): прод-режим, требует применённой миграции (таблицы toponym_*).
  Запись идёт сервисным ключом (service_role) через REST.
"""
from __future__ import annotations

import os

import pandas as pd

OUTDIR = "data/analytics"

# допустимые значения geocode_source в схеме (CHECK); «canonical» курируемый -> manual
_ALLOWED_GEO_SRC = {"gazetteer", "nominatim", "manual"}

# имя таблицы -> имя CSV-файла
CSV_NAMES = {
    "toponyms": "toponyms.csv",
    "toponym_mentions": "toponym_mentions.csv",
    "toponym_syntax": "toponym_syntax.csv",
    "toponym_tonality": "toponym_tonality.csv",
    "toponym_topics": "toponym_topics.csv",
    "toponym_verbs": "toponym_verbs.csv",
}


def write_csv(tables: dict[str, pd.DataFrame], outdir: str = OUTDIR) -> list[str]:
    """Сохраняет таблицы в CSV (utf-8-sig для Excel). Возвращает пути."""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for name, df in tables.items():
        path = os.path.join(outdir, CSV_NAMES.get(name, f"{name}.csv"))
        df.to_csv(path, index=False, encoding="utf-8-sig")
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Supabase (REST). Требует применённой миграции sql/migrate_add_analytics.sql.
# ---------------------------------------------------------------------------

def _client():
    from database.supabase_client import SupabaseClient
    return SupabaseClient().client


def _clean(v):
    """JSON-safe значение: NaN/NaT/None -> None, Timestamp -> ISO, numpy-скаляр -> python."""
    if pd.isna(v):
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if hasattr(v, "item"):
        return v.item()
    return v


def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    """DataFrame -> list[dict] (JSON-safe), только нужные колонки."""
    use = [c for c in cols if c in df.columns]
    recs = df[use].to_dict("records")
    return [{k: _clean(v) for k, v in r.items()} for r in recs]


def _chunks(items: list, size: int = 500):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _select_all(client, table: str, columns: str) -> list:
    """Постранично читает все строки таблицы (PostgREST отдаёт max ~1000 за запрос)."""
    # листаем до пустого ответа, сдвигаясь на фактически вернувшееся число строк
    rows, offset, page = [], 0, 1000
    while True:
        batch = client.table(table).select(columns).range(offset, offset + page - 1).execute().data
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    return rows


def write_supabase(toponyms: pd.DataFrame, mentions: pd.DataFrame, syntax: pd.DataFrame,
                   client=None, processed_post_ids=None) -> dict:
    """Пишет toponyms / toponym_mentions / toponym_syntax в Supabase.

    Шаги: upsert toponyms (по name) -> карта name->id -> вставка mentions/syntax с toponym_id.
    Для затронутых постов старые строки mentions/syntax удаляются (идемпотентность ре-прогона).

    Колонки на входе:
      toponyms : name, display_name, lat, lon, geocode_source
      mentions : post_id, toponym(name), source_type, published_at, sentence_idx,
                 sentence, word, sentence_sentiment, topic_category
      syntax   : post_id, toponym(name), word, normal_form, pos, deprel
    """
    client = client or _client()

    # 1) upsert справочника топонимов (geocode_source приводим к допустимым значениям схемы)
    topo = toponyms.copy()
    has_coords = topo["lat"].notna() & topo["lon"].notna()
    if "geocode_source" in topo.columns:
        gs = topo["geocode_source"].where(topo["geocode_source"].isin(_ALLOWED_GEO_SRC), "manual")
        topo["geocode_source"] = gs.where(has_coords, None)  # без координат -> NULL, не 'manual'
    # строки без координат пишем только name/display_name, чтобы не затереть lat/lon в NULL
    for batch in _chunks(_records(topo[has_coords], ["name", "display_name", "lat", "lon", "geocode_source"])):
        client.table("toponyms").upsert(batch, on_conflict="name").execute()
    for batch in _chunks(_records(topo[~has_coords], ["name", "display_name"])):
        client.table("toponyms").upsert(batch, on_conflict="name").execute()

    # 2) карта name -> id (постранично: топонимов может быть >1000)
    name2id = {r["name"]: r["id"] for r in _select_all(client, "toponyms", "id,name")}

    # 3) чистим старые строки по всем обработанным постам (идемпотентность ре-прогона)
    if processed_post_ids is not None:
        post_ids = sorted({str(p) for p in processed_post_ids if p is not None})
    else:
        post_ids = sorted(set(mentions["post_id"].dropna().astype(str)) |
                          set(syntax["post_id"].dropna().astype(str)))
    for batch in _chunks(post_ids, 100):  # .in_() кладёт id в URL — держим чанк небольшим
        client.table("toponym_mentions").delete().in_("post_id", batch).execute()
        client.table("toponym_syntax").delete().in_("post_id", batch).execute()

    # 4) вставка mentions
    m = mentions.copy()
    m["toponym_id"] = m["toponym"].map(name2id)
    m["toponym_name"] = m["toponym"]
    m = m[m["toponym_id"].notna()]
    m = m.drop_duplicates(subset=["post_id", "sentence_idx", "toponym_id"])
    m_cols = ["post_id", "toponym_id", "toponym_name", "source_type", "published_at",
              "sentence_idx", "sentence", "word", "sentence_sentiment", "topic_category"]
    n_m = 0
    for batch in _chunks(_records(m, m_cols)):
        client.table("toponym_mentions").insert(batch).execute()
        n_m += len(batch)

    # 5) вставка syntax
    s = syntax.copy()
    s["toponym_id"] = s["toponym"].map(name2id)
    s["toponym_name"] = s["toponym"]
    s = s[s["toponym_id"].notna()]
    s_cols = ["post_id", "toponym_id", "toponym_name", "word", "normal_form", "pos", "deprel",
              "sentence_sentiment"]
    n_s = 0
    for batch in _chunks(_records(s, s_cols)):
        client.table("toponym_syntax").insert(batch).execute()
        n_s += len(batch)

    return {"toponyms": len(name2id), "mentions": n_m, "syntax": n_s}


def mark_posts_analyzed(post_ids, client=None) -> int:
    """Проставляет posts.analyzed_at = now() обработанным постам (инкрементальный режим)."""
    from datetime import datetime, timezone

    client = client or _client()
    ts = datetime.now(timezone.utc).isoformat()
    ids = [str(p) for p in post_ids if p is not None]
    n = 0
    for batch in _chunks(ids, 100):  # .in_() кладёт id в URL — держим чанк небольшим
        client.table("posts").update({"analyzed_at": ts}).in_("id", batch).execute()
        n += len(batch)
    return n
