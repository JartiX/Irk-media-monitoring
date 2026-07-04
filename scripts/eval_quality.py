# -*- coding: utf-8 -*-
"""
Оценка точности извлечения топонимов и геокодинга (precision / recall / F1 на ручной разметке).

Шаги:
  1) Сгенерировать шаблон для разметки (выбирает предложения-кандидаты):
        .venv\\Scripts\\python.exe scripts/eval_quality.py --sample 150
     -> data/analytics/eval_gold.csv

  2) Открыть в Excel и заполнить (по каждому предложению):
        missed  — топонимы, которые система ПРОПУСТИЛА (через ;)
        wrong   — ЛИШНИЕ/ошибочные из auto_toponyms (через ;)
        geo_ok  — координата верна? 1=да, 0=нет, пусто=не проверяли

  3) Посчитать метрики:
        .venv\\Scripts\\python.exe scripts/eval_quality.py --score data/analytics/eval_gold.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

GOLD = "data/analytics/eval_gold.csv"


def sample(n: int):
    from analytics import load_vk_xlsx, sentences_frame
    from analytics.analyze import analyze_sentence

    posts = load_vk_xlsx()
    sents = sentences_frame(posts)
    # предложения средней длины — вероятнее с топонимами и контекстом
    cand = sents[sents["sentence"].str.len().between(40, 300)].reset_index(drop=True)
    step = max(1, len(cand) // n)  # детерминированно: каждый step-й (без random)
    pick = cand.iloc[::step].head(n)

    rows = []
    for s in pick["sentence"]:
        mentions, _ = analyze_sentence(s)
        autos = ";".join(sorted({m["toponym"] for m in mentions}))
        rows.append({"sentence": s, "auto_toponyms": autos, "missed": "", "wrong": "", "geo_ok": ""})

    os.makedirs(os.path.dirname(GOLD), exist_ok=True)
    pd.DataFrame(rows).to_csv(GOLD, index=False, encoding="utf-8-sig")
    print(f"Шаблон: {GOLD} ({len(rows)} предложений).")
    print("Заполни столбцы missed / wrong / geo_ok в Excel и запусти --score.")


def _split(v) -> set:
    if not isinstance(v, str) or not v.strip():
        return set()
    return {x.strip().lower() for x in v.replace(",", ";").split(";") if x.strip()}


def score(path: str):
    df = pd.read_csv(path).fillna("")
    tp = fp = fn = 0
    geo_ok = geo_tot = 0
    for _, r in df.iterrows():
        auto = _split(r.get("auto_toponyms"))
        wrong = _split(r.get("wrong")) & auto
        missed = _split(r.get("missed"))
        tp += len(auto - wrong)
        fp += len(wrong)
        fn += len(missed)
        g = str(r.get("geo_ok", "")).strip()
        if g in ("1", "0"):
            geo_tot += 1
            geo_ok += int(g == "1")

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"NER топонимов:  TP={tp}  FP={fp}  FN={fn}")
    print(f"  precision = {prec:.2f}   recall = {rec:.2f}   F1 = {f1:.2f}")
    if geo_tot:
        print(f"Геокодинг: точность {geo_ok}/{geo_tot} = {geo_ok / geo_tot:.2f}")
    else:
        print("Геокодинг: столбец geo_ok не заполнен (1/0) — пропускаю.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Оценка качества NER топонимов и геокодинга")
    ap.add_argument("--sample", type=int, help="сгенерировать шаблон на N предложений")
    ap.add_argument("--score", type=str, help="посчитать метрики по размеченному CSV")
    a = ap.parse_args()
    if a.sample:
        sample(a.sample)
    elif a.score:
        score(a.score)
    else:
        ap.print_help()
