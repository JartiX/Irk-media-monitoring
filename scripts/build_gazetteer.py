# -*- coding: utf-8 -*-
"""
Строит большой газеттир топонимов Иркутской области из OpenStreetMap (Overpass API):
населённые пункты + природные объекты (пики/заливы/мысы/острова/источники) с координатами.

Запуск: .venv\\Scripts\\python.exe scripts/build_gazetteer.py
Результат: data/osm_gazetteer.csv (name, lat, lon, type)
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import requests

OUT = "data/osm_gazetteer.csv"
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
QUERY = """
[out:json][timeout:180];
area["name"="Иркутская область"]["admin_level"="4"]->.a;
(
  node["place"~"^(city|town|village|hamlet)$"]["name"](area.a);
  node["natural"~"^(peak|bay|cape|spring|island)$"]["name"](area.a);
  way["natural"="water"]["name"](area.a);
  relation["natural"="water"]["name"](area.a);
  relation["natural"~"^(bay|island)$"]["name"](area.a);
);
out center tags;
"""


def fetch():
    """Возвращает JSON Overpass или бросает исключение (в т.ч. на HTTP-200 с remark/пустым ответом)."""
    last = "—"
    for url in ENDPOINTS:
        try:
            print("Overpass:", url)
            r = requests.post(url, data={"data": QUERY}, timeout=200,
                              headers={"User-Agent": "irk-media-monitoring/0.1"})
            r.raise_for_status()
            j = r.json()
            # таймаут Overpass приходит как HTTP 200 с пустым elements — это ошибка, а не успех
            if not j.get("elements"):
                raise RuntimeError(f"пусто (remark: {str(j.get('remark'))[:80]})")
            if j.get("remark"):
                print("  remark (не фатально):", str(j.get("remark"))[:80])
            return j
        except Exception as e:
            last = str(e)[:120]
            print("  fail:", last)
    raise RuntimeError(f"все Overpass-эндпоинты недоступны: {last}")


def main():
    data = fetch()
    rows, seen = [], set()
    for el in data.get("elements", []):
        t = el.get("tags", {})
        name = t.get("name")
        if not name:
            continue
        lat = el.get("lat", (el.get("center") or {}).get("lat"))
        lon = el.get("lon", (el.get("center") or {}).get("lon"))
        if lat is None or lon is None:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        # тип: НП -> place (city/town/village); водоём -> water (lake/reservoir/river); иначе natural
        rows.append((name, lat, lon, t.get("place") or t.get("water") or t.get("natural") or "water"))
    if len(rows) < 100:   # подозрительно мало — не перезаписываем файл
        raise SystemExit(f"Overpass вернул лишь {len(rows)} объектов — похоже на сбой; CSV не перезаписан")
    os.makedirs("data", exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "lat", "lon", "type"])
        w.writerows(rows)
    os.replace(tmp, OUT)   # атомарная замена — частичная запись не повредит прежний файл
    print(f"сохранено {len(rows)} объектов -> {OUT}")
    from collections import Counter
    print("по типам:", dict(Counter(r[3] for r in rows).most_common(8)))


if __name__ == "__main__":
    main()
