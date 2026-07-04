"""
Большой газеттир топонимов Иркутской области из OSM (data/osm_gazetteer.csv).

Строится скриптом scripts/build_gazetteer.py (Overpass API): НП + природные объекты с координатами.
Используется для (1) оффлайн-геокодинга и (2) повышения полноты извлечения (словарный матчинг).
"""
from __future__ import annotations

import os

import pandas as pd

from .toponyms import normalize_toponym

OSM_PATH = "data/osm_gazetteer.csv"

_index = None
_names = None


def osm_index() -> dict:
    """{нормализованное имя -> (lat, lon, display_name, type)}. Пусто, если файла нет (напр. в CI)."""
    global _index, _names
    if _index is None:
        _index = {}
        if os.path.exists(OSM_PATH):
            try:
                df = pd.read_csv(OSM_PATH)
            except Exception:
                df = None
            if df is not None:
                for _, r in df.iterrows():
                    try:                       # пропускаем битые строки
                        if pd.isna(r["name"]) or pd.isna(r["lat"]) or pd.isna(r["lon"]):
                            continue
                        name = str(r["name"]).strip()
                        lat, lon = float(r["lat"]), float(r["lon"])
                    except (ValueError, KeyError, TypeError):
                        continue
                    if not name:
                        continue
                    key = normalize_toponym(name)
                    if key and key not in _index:
                        _index[key] = (lat, lon, name, str(r.get("type", "")))
        _names = set(_index.keys())
    return _index


def osm_names() -> set:
    """Множество нормализованных имён (для быстрого словарного матчинга)."""
    if _names is None:
        osm_index()
    return _names
