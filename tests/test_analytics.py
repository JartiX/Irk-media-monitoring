# -*- coding: utf-8 -*-
"""
Регрессионные тесты слоя аналитики.

Запуск: .venv\\Scripts\\python.exe tests/test_analytics.py   (или pytest tests/test_analytics.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.toponyms import normalize_toponym
from analytics.sentiment import sentence_sentiment
from analytics.topics import classify
from analytics.geocode import geocode_toponym
from analytics.analyze import analyze_sentence


def test_normalize():
    assert normalize_toponym("Байкала") == "байкал"
    assert normalize_toponym("Ольхоне") == "ольхон"


def test_sentiment_lexicon():
    assert sentence_sentiment("прекрасно, рекомендую") > 0
    assert sentence_sentiment("ужасно, грязно и дорого") < 0
    assert sentence_sentiment("обычный рабочий день") == 0


def test_topics():
    assert classify("рафтинг и треккинг по горам") == "активный"
    assert classify("дегустация местной кухни") == "гастрономический"
    assert classify("массаж и грязелечение в санатории") == "оздоровительный"
    assert classify("текст без туристической темы") is None


def test_sentiment_no_false_negatives():
    # омонимы не должны метиться негативом
    assert sentence_sentiment("дорогой друг приехал на Байкал") == 0      # «дорогой» != цена
    assert sentence_sentiment("наглядный пример маршрута") == 0           # «наглядный» != «наглый»
    assert sentence_sentiment("новый маршрут запущен") == 0               # «запущен» = launched
    assert sentence_sentiment("грубо говоря, это рядом") == 0             # вводное «грубо говоря»
    # истинные негативы сохраняются
    assert sentence_sentiment("дороговизна билетов отпугивает") < 0
    assert sentence_sentiment("запущенный заброшенный парк") < 0
    assert sentence_sentiment("наглый таксист обманул") < 0


def test_topics_no_false_positives():
    assert classify("красивая поза лотоса для фото") != "гастрономический"  # «поза» != блюдо
    assert classify("резкий подъём цен в магазине") != "активный"          # «подъём цен» != туризм
    assert classify("буузы и омуль в кафе") == "гастрономический"           # блюдо ловим по «бууз»
    assert classify("сплав по горной реке") == "активный"                  # туристический сплав
    assert classify("я сплавал в бассейне") != "активный"                  # глагол «сплавал» != сплав


def test_topics_splav_per_token():
    # пер-токенный путь (analyze): «сплав» во всех падежах через лемму -> активный
    m, _ = analyze_sentence("На Байкале популярен сплав по рекам.")
    assert any(x["topic_category"] == "активный" for x in m), m


def test_geocode_canonical_baikal_on_lake():
    g = geocode_toponym("байкал")
    assert g and g["in_region"]
    assert 53.0 < g["lat"] < 54.5 and 107.0 < g["lon"] < 109.5  # озеро, не стадион в Иркутске


def test_empty_and_whitespace():
    assert analyze_sentence("") == ([], [])
    assert analyze_sentence("   \n\t ") == ([], [])
    assert analyze_sentence("...") == ([], [])


def test_per_toponym_attribution():
    m, _ = analyze_sentence("Байкал прекрасен, но в Листвянке грязно и дорого")
    d = {x["toponym"]: x["sentence_sentiment"] for x in m}
    assert d.get("байкал", 0) > 0, d
    assert d.get("листвянка", 0) < 0, d


def test_fp_filters():
    # животное/наречие не топоним
    assert all(x["toponym"] != "нерпа" for x in analyze_sentence("Нерпа уплыла от дайвера.")[0])
    # подстрока внутри слова не даёт фантома
    toks = {x["toponym"] for x in analyze_sentence("Прибайкалье и Забайкалье разные.")[0]}
    assert "байкал" not in toks


def test_verb_link_filters_light_verbs():
    _, links = analyze_sentence("Съездили на Ольхон искупаться, Байкал стал чище.")
    verbs = {x["normal_form"] for x in links if x["pos"] == "VERB"}
    assert "стать" not in verbs           # служебный глагол отфильтрован
    assert "съездить" in verbs or "искупаться" in verbs


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print("OK", fn.__name__)
    print(f"\nВСЕ {len(tests)} ТЕСТОВ ПРОЙДЕНЫ")


if __name__ == "__main__":
    import sys as _s
    _s.stdout.reconfigure(encoding="utf-8")
    _run_all()
