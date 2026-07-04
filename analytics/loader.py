"""
Загрузка постов из источников в единый DataFrame.

Унифицированная схема:
    post_uid     - уникальный ключ поста в рамках выборки (str)
    post_id      - id поста в исходной системе (str)
    source       - тип источника: 'vk' | 'news' | 'telegram' | 'supabase'
    group_id     - id группы/источника (str|None)
    group_name   - человекочитаемое имя группы (str|None)
    published_at - дата публикации (datetime|NaT)
    text         - текст поста (str)
"""
from __future__ import annotations

import glob
import os
import re

import pandas as pd

VK_DIR = "data/VK"
GAZETTEER_PATH = "data/Irk_obl_sights.csv"

SCHEMA = ["post_uid", "post_id", "source", "group_id", "group_name", "published_at", "text"]

# Имя файла VK-выгрузки: 'baikal(-12377).xlsx' -> ('baikal', '-12377')
_VK_FNAME = re.compile(r"(.+?)\((-?\d+)\)\.xlsx$")


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEMA)


def load_vk_xlsx(vk_dir: str = VK_DIR) -> pd.DataFrame:
    """Читает VK-выгрузки *.xlsx из каталога в единый DataFrame."""
    frames = []
    for path in sorted(glob.glob(os.path.join(vk_dir, "*.xlsx"))):
        fname = os.path.basename(path)
        m = _VK_FNAME.match(fname)
        group_name = m.group(1) if m else os.path.splitext(fname)[0]
        group_id = m.group(2) if m else None

        df = pd.read_excel(path)
        if "text" not in df.columns or "id" not in df.columns:
            continue
        out = pd.DataFrame({
            "post_id": df["id"].astype(str),
            "source": "vk",
            "group_id": group_id,
            "group_name": group_name,
            "published_at": pd.to_datetime(df.get("date"), errors="coerce"),
            "text": df["text"].astype("string"),
        })
        frames.append(out)

    if not frames:
        return _empty()

    res = pd.concat(frames, ignore_index=True)
    res = res[res["text"].notna() & (res["text"].str.strip().str.len() > 0)].reset_index(drop=True)
    # id уникален внутри группы -> ключ = group_id + post_id
    res["post_uid"] = res["group_id"].fillna("") + "_" + res["post_id"]
    return res[SCHEMA]


def load_gazetteer(path: str = GAZETTEER_PATH) -> pd.DataFrame:
    """Читает газеттир достопримечательностей (UTF-16, tab, десятичная запятая).

    Возвращает колонки: name, lat, lon, type, address (lat/lon как float).
    """
    df = None
    for enc in ("utf-16", "utf-16-le", "utf-8-sig", "cp1251"):
        try:
            tmp = pd.read_csv(path, sep="\t", encoding=enc)
            cols = {str(c).strip().lower() for c in tmp.columns}
            if "name" in cols and ("lat" in cols or "lon" in cols):
                df = tmp
                break
        except Exception:
            continue
    if df is None:
        raise RuntimeError(f"Не удалось прочитать газеттир (нет колонок name/lat/lon): {path}")

    df.columns = [c.strip() for c in df.columns]
    for col in ("lat", "lon"):
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
    keep = [c for c in ("name", "lat", "lon", "type", "address") if c in df.columns]
    df = df[keep].copy()
    df = df[df["name"].notna()]
    df["name"] = df["name"].astype(str).str.strip()
    return df.reset_index(drop=True)


def load_supabase_posts(limit: int = 1000, only_unanalyzed: bool = False,
                        relevant_only: bool = False) -> pd.DataFrame:
    """Читает посты из Supabase в унифицированную схему (прод-режим).

    only_unanalyzed=True -> только посты с analyzed_at IS NULL (инкрементальный режим).
    Отредактированные посты инкрементом не берутся — их переобрабатывает полный прогон.
    Ленивый импорт supabase-клиента, чтобы оффлайн-режимы не требовали окружения.
    """
    from database.supabase_client import SupabaseClient

    client = SupabaseClient().client

    def _query():
        # тип/имя источника по FK posts.source_id
        q = client.table("posts").select("*, sources(type, name)")
        if only_unanalyzed:
            q = q.is_("analyzed_at", "null")
        if relevant_only:
            q = q.eq("is_relevant", True)
        # вторичный ключ для стабильного порядка страниц
        return q.order("created_at", desc=True).order("id", desc=True)

    # пагинация: PostgREST отдаёт максимум ~1000 строк за запрос
    rows, offset = [], 0
    while len(rows) < limit:
        cur = min(1000, limit - len(rows))
        batch = _query().range(offset, offset + cur - 1).execute().data
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    rows = rows[:limit]
    df = pd.DataFrame(rows)
    if df.empty:
        return _empty()

    title = df.get("title")
    content = df.get("content")
    text = (
        (title.fillna("") if title is not None else "")
        + ". "
        + (content.fillna("") if content is not None else "")
    )

    def _src(field):
        s = df.get("sources")
        if s is None:
            return None
        return s.map(lambda x: x.get(field) if isinstance(x, dict) else None)

    src_type = _src("type")
    src_name = _src("name")
    out = pd.DataFrame({
        "post_id": df["id"].astype(str),
        "post_uid": df["id"].astype(str),
        "source": (src_type.fillna("unknown") if src_type is not None else "supabase"),
        "group_id": df.get("source_id"),
        "group_name": src_name,
        "published_at": pd.to_datetime(df.get("published_at"), errors="coerce"),
        "text": text.astype("string"),
    })
    return out[SCHEMA]
