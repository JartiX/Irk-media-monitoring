# -*- coding: utf-8 -*-
"""
Демонстрация: VK-посты -> предложения -> топонимы.

Запуск из корня репозитория:
    .venv\\Scripts\\python.exe scripts\\analytics_demo.py            # на всех данных (~25с)
    .venv\\Scripts\\python.exe scripts\\analytics_demo.py --limit 2000   # быстрее, на подвыборке

Результаты сохраняются в data/analytics/ (CSV в кодировке для Excel):
    demo_toponym_freq.csv  - топонимы по частоте упоминаний
    demo_mentions.csv      - упоминания: топоним + предложение + пост
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from analytics import load_vk_xlsx, sentences_frame, extract_toponyms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="ограничить число предложений (0 = все)")
    args = ap.parse_args()

    os.makedirs("data/analytics", exist_ok=True)

    print("1) Загрузка VK-постов...")
    posts = load_vk_xlsx()
    print(f"   постов: {len(posts)} из {posts['group_name'].nunique()} групп")

    print("2) Сегментация на предложения...")
    sents = sentences_frame(posts)
    if args.limit:
        sents = sents.head(args.limit)
    print(f"   предложений: {len(sents)}")

    print(f"3) Извлечение топонимов (ориентир ~{max(1, round(len(sents) * 1.3 / 1000))}с)...")
    t0 = time.time()
    mentions = extract_toponyms(sents)
    print(f"   упоминаний: {len(mentions)} за {time.time() - t0:.0f}с")

    freq = (
        mentions.groupby("toponym")
        .agg(n_repeat=("toponym", "size"), sample_word=("word", "first"))
        .sort_values("n_repeat", ascending=False)
        .reset_index()
    )

    freq.to_csv("data/analytics/demo_toponym_freq.csv", index=False, encoding="utf-8-sig")
    mentions.head(3000).to_csv("data/analytics/demo_mentions.csv", index=False, encoding="utf-8-sig")

    print(f"\n=== ТОП-20 топонимов (всего уникальных: {len(freq)}) ===")
    print(freq.head(20).to_string(index=False))

    print("\n=== Примеры: [топоним] <- предложение ===")
    for _, r in mentions.head(10).iterrows():
        print(f"  [{r['toponym']}] <- {str(r['sentence'])[:80]}")

    print("\nОткрой в Excel:")
    print("  data/analytics/demo_toponym_freq.csv   (частоты топонимов)")
    print("  data/analytics/demo_mentions.csv       (упоминания + предложения)")


if __name__ == "__main__":
    main()
