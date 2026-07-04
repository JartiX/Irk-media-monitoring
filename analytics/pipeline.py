"""
Оркестрация аналитического конвейера.

Цепочка: загрузка -> предложения -> топонимы(+STOP/канон) -> тональность+тематика ->
синтаксис -> геокодинг -> итоговые таблицы -> CSV (и/или Supabase).

Запуск из корня репозитория:
    .venv\\Scripts\\python.exe -m analytics.pipeline --source vk --limit 6000
    .venv\\Scripts\\python.exe -m analytics.pipeline --source vk --to-db   (после миграции БД)
"""
from __future__ import annotations

import argparse
import time

from . import loader, preprocess, storage
from .analyze import analyze_frame
from .sentiment import tonality_by_toponym
from .geocode import geocode_frame
from .canonical_geo import CANONICAL_TOPONYMS, STOP_TOPONYMS


def _canonical(norm: str) -> str:
    """Каноническое имя места (объединяет варианты формы) или нормализованная форма."""
    c = CANONICAL_TOPONYMS.get(norm)
    return c[0] if c else norm


def run(source: str = "vk", sentence_limit: int = 0, geocode_top: int = 120,
        to_db: bool = False, incremental: bool = False, post_limit: int = 5000,
        ml_sentiment: bool = False, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    from . import analyze as _analyze
    _analyze.USE_ML_SENTIMENT = bool(ml_sentiment)
    if ml_sentiment:
        log("[ml] тональность: ML-модель (предложения с 1 топонимом)")
    if incremental and not to_db and source != "vk":
        log("[!] --incremental без --to-db: посты не помечаются (CSV-режим)")

    # 1) загрузка + сегментация
    if source == "vk":
        posts = loader.load_vk_xlsx()
    else:
        posts = loader.load_supabase_posts(limit=post_limit, only_unanalyzed=incremental)
    sents = preprocess.sentences_frame(posts)
    if sentence_limit and not incremental:
        sents = sents.head(sentence_limit)
    log(f"[1] постов={len(posts)} предложений={len(sents)}")
    if len(sents) == 0:
        log("[!] нет предложений (возможно, все посты уже обработаны)")
        return {}

    # 2-4) ЕДИНЫЙ разбор: топонимы + тональность/тема НА ТОПОНИМ + синтаксис (один проход)
    mentions, syntax = analyze_frame(sents)
    mentions = mentions[~mentions["toponym"].isin(STOP_TOPONYMS)].reset_index(drop=True)
    mentions["place"] = mentions["toponym"].map(_canonical)
    syntax = syntax[~syntax["toponym"].isin(STOP_TOPONYMS)].reset_index(drop=True)
    syntax["place"] = syntax["toponym"].map(_canonical)
    log(f"[2-4] упоминаний={len(mentions)} мест={mentions['place'].nunique()} синт.связей={len(syntax)}")

    # 5) геокодинг топ-N топонимов по частоте
    freq = (mentions.groupby("toponym").size().reset_index(name="n")
            .sort_values("n", ascending=False))
    freq["sample_word"] = freq["toponym"].map(mentions.groupby("toponym")["word"].first())
    geo = geocode_frame(freq, limit=geocode_top).rename(columns={"source": "geocode_source"})
    log(f"[5] геокодировано {int(geo['lat'].notna().sum())} из {len(geo)} топонимов "
        f"(Nominatim-лимит топ-{geocode_top}; канон/газеттир — для всех)")
    mentions = mentions.merge(
        geo[["toponym", "lat", "lon", "in_region", "geocode_source"]], on="toponym", how="left")

    # 6) итоговые таблицы (по каноническому месту)
    toponyms_tbl = (mentions.groupby("place")
                    .agg(n_repeat=("place", "size"), lat=("lat", "first"), lon=("lon", "first"),
                         in_region=("in_region", "first"), geocode_source=("geocode_source", "first"))
                    .reset_index().rename(columns={"place": "name"}))
    toponyms_tbl["display_name"] = toponyms_tbl["name"]
    toponyms_tbl = toponyms_tbl.sort_values("n_repeat", ascending=False)

    tonality_tbl = tonality_by_toponym(mentions, toponym_col="place").rename(columns={"place": "name"})
    topics_tbl = (mentions[mentions["topic_category"] != ""]
                  .groupby(["place", "topic_category"]).size().reset_index(name="cnt")
                  .sort_values("cnt", ascending=False))
    verbs_tbl = (syntax[syntax["pos"] == "VERB"]
                 .groupby(["place", "normal_form"]).size().reset_index(name="cnt")
                 .sort_values("cnt", ascending=False))

    tables = {
        "toponyms": toponyms_tbl,
        "toponym_mentions": mentions,
        "toponym_syntax": syntax,
        "toponym_tonality": tonality_tbl,
        "toponym_topics": topics_tbl,
        "toponym_verbs": verbs_tbl,
    }
    paths = storage.write_csv(tables)
    log(f"[6] CSV: {len(paths)} файлов в {storage.OUTDIR}/")

    # 7) запись в Supabase (опционально; после применения миграции)
    if to_db:
        topo_db = toponyms_tbl[["name", "display_name", "lat", "lon", "geocode_source"]]
        # place -> toponym (каноническое имя как идентичность); старый нормализованный toponym убираем
        m_db = (mentions.drop(columns=["toponym"])
                .rename(columns={"place": "toponym", "source": "source_type"}))
        s_db = syntax.drop(columns=["toponym"]).rename(columns={"place": "toponym"})
        res = storage.write_supabase(topo_db, m_db, s_db,
                                     processed_post_ids=posts["post_id"].tolist())
        log(f"[7] Supabase: {res}")
        if incremental and source != "vk":
            n_marked = storage.mark_posts_analyzed(posts["post_id"].tolist())
            log(f"[7] помечено analyzed_at: {n_marked} постов")

    log(f"готово за {time.time() - t0:.0f}с")
    return tables


def main():
    ap = argparse.ArgumentParser(description="Аналитический конвейер топонимов")
    ap.add_argument("--source", default="vk", choices=["vk", "supabase"])
    ap.add_argument("--limit", type=int, default=0, help="ограничить число предложений (0=все)")
    ap.add_argument("--geocode-top", type=int, default=120, help="сколько топ-топонимов геокодировать")
    ap.add_argument("--to-db", action="store_true", help="писать результаты в Supabase")
    ap.add_argument("--incremental", action="store_true",
                    help="только необработанные посты (analyzed_at IS NULL), затем пометить их")
    ap.add_argument("--post-limit", type=int, default=5000, help="макс. постов из Supabase за прогон")
    ap.add_argument("--ml-sentiment", action="store_true", help="ML-тональность (точнее, но медленнее)")
    a = ap.parse_args()
    run(source=a.source, sentence_limit=a.limit, geocode_top=a.geocode_top,
        to_db=a.to_db, incremental=a.incremental, post_limit=a.post_limit,
        ml_sentiment=a.ml_sentiment)


if __name__ == "__main__":
    main()
