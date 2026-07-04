"""
Геокодинг топонимов.

Схема: курируемый канон -> локальный газеттир (оффлайн) -> Nominatim (OpenStreetMap)
с ограничением частоты запросов и кэшем на диске. Флаг in_region по bbox области.
Nominatim вызывается только для топ-N топонимов по частоте — длинный хвост берём оффлайн.
"""
from __future__ import annotations

import json
import os

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from .loader import load_gazetteer
from .toponyms import normalize_toponym
from .canonical_geo import CANONICAL_TOPONYMS
from .osm_gazetteer import osm_index as _osm_index

CACHE_PATH = "data/analytics/geocode_cache.json"
USER_AGENT = "irk-media-monitoring-analytics/0.1"
# bbox: (lat_min, lat_max, lon_min, lon_max) — Иркутская область + Байкал
REGION_BBOX = (51.0, 64.0, 96.0, 119.0)

_NULL = {"lat": None, "lon": None, "source": None, "display_name": None, "in_region": None}

# типы-омонимы газеттира (стадион/музей/кинотеатр/…) — не годятся как координата топонима
_GAZ_POI_BLACKLIST = (
    "стадион", "sport", "stadium", "музей", "museum", "кинотеатр", "cinema", "театр", "theatre",
    "развлека", "entertain", "бассейн", "памятник", "monument", "мемориал", "memorial",
    "sculpt", "historic", "religion", "храм", "церков", "church", "монастыр",
    "отель", "hotel", "ресторан", "кафе", "магазин", "shop",
)

_geocode_limited = None
_gaz_index = None
_cache = None


def _region_check(lat: float, lon: float) -> bool:
    la0, la1, lo0, lo1 = REGION_BBOX
    return (la0 <= lat <= la1) and (lo0 <= lon <= lo1)


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        raw = {}
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH, encoding="utf-8") as f:
                    raw = json.load(f)
            except (json.JSONDecodeError, OSError):
                raw = {}                    # повреждённый файл -> пустой кэш
            if not isinstance(raw, dict):
                raw = {}
        # канонические ключи берём из канона, не из кэша
        _cache = {k: v for k, v in raw.items() if k not in CANONICAL_TOPONYMS}
    return _cache


def _save_cache() -> None:
    if _cache is None:   # кэш не загружался — нечего сохранять
        return
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_cache, f, ensure_ascii=False, indent=0)
    os.replace(tmp, CACHE_PATH)  # атомарная замена


def _gazetteer_index() -> dict:
    """{нормализованное имя -> (lat, lon, оригинальное имя)} по газеттиру."""
    global _gaz_index
    if _gaz_index is None:
        try:
            gaz = load_gazetteer()
        except Exception:
            _gaz_index = {}  # газеттир недоступен — работаем на каноне + Nominatim
            return _gaz_index
        idx = {}
        for _, r in gaz.iterrows():
            if pd.isna(r["lat"]) or pd.isna(r["lon"]):
                continue
            typ = str(r.get("type", "")).lower()
            if any(b in typ for b in _GAZ_POI_BLACKLIST):
                continue
            key = normalize_toponym(str(r["name"]))
            if key and key not in idx:
                idx[key] = (float(r["lat"]), float(r["lon"]), str(r["name"]))
        _gaz_index = idx
    return _gaz_index


def _nominatim():
    global _geocode_limited
    if _geocode_limited is None:
        geo = Nominatim(user_agent=USER_AGENT, timeout=15)
        _geocode_limited = RateLimiter(geo.geocode, min_delay_seconds=1.1, swallow_exceptions=False)
    return _geocode_limited


def geocode_toponym(toponym: str, query_hint: str | None = None, use_nominatim: bool = True) -> dict | None:
    """Геокодирует один топоним. Кэш по нормализованному имени (toponym)."""
    # 0) курируемый канон региона — приоритет
    canon = CANONICAL_TOPONYMS.get(toponym)
    if canon:
        name, lat, lon = canon
        return {"lat": lat, "lon": lon, "source": "canonical",
                "display_name": name, "in_region": _region_check(lat, lon)}

    # 1) OSM-газеттир области (оффлайн) — до кэша
    oi = _osm_index().get(toponym)
    if oi:
        lat, lon, name = oi[0], oi[1], oi[2]
        return {"lat": lat, "lon": lon, "source": "gazetteer",
                "display_name": name, "in_region": _region_check(lat, lon)}

    cache = _load_cache()
    if toponym in cache:
        return cache[toponym]

    result = None

    # 2) газеттир достопримечательностей (POI-фильтрованный)
    gi = _gazetteer_index().get(toponym)
    if gi:
        lat, lon, name = gi
        result = {"lat": lat, "lon": lon, "source": "gazetteer",
                  "display_name": name, "in_region": _region_check(lat, lon)}
    # 3) Nominatim, ограниченный регионом (viewbox + bounded)
    elif use_nominatim:
        geocode = _nominatim()
        q = (query_hint if isinstance(query_hint, str) and query_hint.strip() else toponym).strip()
        vb = [(REGION_BBOX[1], REGION_BBOX[2]), (REGION_BBOX[0], REGION_BBOX[3])]
        try:
            loc = geocode(f"{q}, Иркутская область, Россия", language="ru",
                          country_codes="ru", viewbox=vb, bounded=True)
            if loc is None:
                loc = geocode(f"{q}, Россия", language="ru",
                              country_codes="ru", viewbox=vb, bounded=True)
        except Exception:
            return None  # ошибка Nominatim — не кэшируем
        if loc is not None:
            result = {"lat": loc.latitude, "lon": loc.longitude, "source": "nominatim",
                      "display_name": loc.address, "in_region": _region_check(loc.latitude, loc.longitude)}

    # None кэшируем только если Nominatim опрашивался
    if result is not None or use_nominatim:
        cache[toponym] = result
    return result


def geocode_frame(df: pd.DataFrame, toponym_col: str = "toponym",
                  hint_col: str = "sample_word", use_nominatim: bool = True,
                  limit: int | None = None) -> pd.DataFrame:
    """Геокодирует уникальные топонимы из df (напр., таблицы частот).

    Канон и газеттир (оффлайн, бесплатно) применяются КО ВСЕМ топонимам;
    лимит ограничивает только запросы к Nominatim (топ-N по порядку df).
    """
    work = df.drop_duplicates(subset=[toponym_col])
    has_hint = hint_col in work.columns
    # топонимы, которым разрешён Nominatim (топ-N по порядку df)
    nom_allowed = (set(work.head(limit)[toponym_col].astype(str))
                   if (limit and use_nominatim) else None)

    rows = []
    try:
        for _, r in work.iterrows():
            top = str(r[toponym_col])
            hint = r[hint_col] if has_hint else None
            allow_nom = use_nominatim and (nom_allowed is None or top in nom_allowed)
            res = geocode_toponym(top, query_hint=hint, use_nominatim=allow_nom)
            rows.append({toponym_col: r[toponym_col], **(res or _NULL)})
    finally:
        _save_cache()
    return pd.DataFrame(rows, columns=[toponym_col, "lat", "lon", "source", "display_name", "in_region"])
