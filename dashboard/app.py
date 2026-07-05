"""
Дашборд: анализ текстовых данных медиапространства туристического сегмента (Прибайкалье).

Запуск из корня репозитория:
    .venv\\Scripts\\streamlit.exe run dashboard/app.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import load_analytics, aggregate, word_stats

st.set_page_config(page_title="Туризм Прибайкалья — медиа-аналитика", layout="wide")

# перерисовка только своего блока при смене виджета внутри (иначе перерисовывается вся страница)
_frag = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)


def fragment(fn):
    return _frag(fn) if _frag else fn


@st.cache_data(ttl=300)
def _load():
    return load_analytics()


def _wordcloud_fig(ws, max_words=140):
    """Фигура облака слов: размер = частота, цвет = тональность (зелёный/красный/серый)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fmg
    try:
        from wordcloud import WordCloud
    except Exception:
        st.warning("Не установлен пакет `wordcloud` (pip install wordcloud) — облако недоступно.")
        return None
    if ws is None or ws.empty:
        return None
    w = ws.groupby("word", as_index=False).agg(freq=("freq", "sum"), sentiment=("sentiment", "mean"))
    w = w[w["freq"] > 0]
    if w.empty:
        return None
    freqs = dict(zip(w["word"].astype(str), w["freq"].astype(float)))
    smap = dict(zip(w["word"].astype(str), w["sentiment"].astype(float)))

    def _color(word, **kwargs):
        s = smap.get(word, 0.0)
        if s > 0.05:
            return "#2e7d32"   # зелёный — позитивный контекст
        if s < -0.05:
            return "#c62828"   # красный — негативный контекст
        return "#9e9e9e"       # серый — нейтрально

    font = fmg.findfont("DejaVu Sans")   # кириллица, кроссплатформенно
    wc = WordCloud(width=1100, height=500, background_color="white", prefer_horizontal=0.92,
                   max_words=max_words, font_path=font, color_func=_color,
                   collocations=False).generate_from_frequencies(freqs)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


mentions, toponyms, syntax, src = _load()

st.title("🗺️ Туристическое медиапространство Прибайкалья")
st.caption(f"Источник данных: **{src}** · всего упоминаний топонимов: {len(mentions)}")

if mentions.empty:
    st.warning("Нет данных. Запусти `python -m analytics.pipeline --source supabase --to-db` "
               "или проверь подключение к Supabase / наличие data/analytics/*.csv.")
    st.stop()

# ----------------------------- Фильтры -----------------------------
with st.sidebar:
    st.header("Фильтры")
    src_opts = sorted(mentions["source_type"].dropna().unique())
    sel_sources = st.multiselect("Источник", src_opts, default=src_opts)

    topic_opts = sorted(t for t in mentions["topic_category"].unique() if t)
    sel_topics = st.multiselect("Вид туризма (тематика)", topic_opts, default=[])

    date_from = date_to = None
    dates = mentions["published_at"].dropna() if "published_at" in mentions else pd.Series([], dtype="datetime64[ns, UTC]")
    if not dates.empty:
        dmin, dmax = dates.min().date(), dates.max().date()
        rng = st.date_input("Период публикации", (dmin, dmax), min_value=dmin, max_value=dmax)
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            date_from = pd.Timestamp(rng[0], tz="UTC")
            date_to = pd.Timestamp(rng[1], tz="UTC") + pd.Timedelta(days=1)

    min_n = st.slider("Мин. упоминаний топонима", 1, 25, 2)
    st.caption("Данные кэшируются на 5 мин. «Rerun» — обновить.")

fm, agg = aggregate(mentions, toponyms, sel_sources or None, date_from, date_to, sel_topics or None)
agg = agg[agg["n"] >= min_n].reset_index(drop=True)

if fm.empty:
    st.info("Под выбранные фильтры нет упоминаний.")
    st.stop()

# ----------------------------- KPI -----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Топонимов", int(agg["toponym_name"].nunique()))
k2.metric("Упоминаний", int(fm.shape[0]))
k3.metric("Источников", int(fm["source_type"].nunique()))
k4.metric("Геокодировано", int(agg["lat"].notna().sum()))

# ----------------------------- Карта -----------------------------
st.subheader("Карта упоминаний · размер = частота, цвет = тональность")
mp = agg.dropna(subset=["lat", "lon"]).copy()
if not mp.empty:
    # sqrt-масштаб с минимальным размером: иначе редкие точки при большом разбросе частот исчезают
    n = mp["n"].astype(float)
    lo, hi = np.sqrt(n.min()), np.sqrt(n.max())
    mp["msize"] = 9 + 27 * (np.sqrt(n) - lo) / (hi - lo + 1e-9)
    fig = go.Figure()
    # чёрная подложка чуть больше цветной точки — контрастная обводка (marker.line у карт нет)
    fig.add_trace(go.Scattermap(
        lat=mp["lat"], lon=mp["lon"], mode="markers",
        marker=dict(size=mp["msize"] + 4, color="black"),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scattermap(
        lat=mp["lat"], lon=mp["lon"], mode="markers",
        marker=dict(size=mp["msize"], color=mp["sentiment"], colorscale="RdYlGn",
                    cmid=0, opacity=1.0, showscale=True,
                    colorbar=dict(title="Тон.", thickness=12)),
        text=mp["toponym_name"],
        customdata=np.stack([mp["n"], mp["sentiment"]], axis=-1),
        hovertemplate="<b>%{text}</b><br>упоминаний: %{customdata[0]:.0f}"
                      "<br>тональность: %{customdata[1]:.0f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(map=dict(style="open-street-map", zoom=4.2,
                               center=dict(lat=54.0, lon=107.5)),
                      margin=dict(l=0, r=0, t=0, b=0), height=520)
    st.plotly_chart(fig, width="stretch")
else:
    st.info("Под фильтр нет геокодированных топонимов.")

# -------------------- Топ топонимов + виды туризма (круговая) --------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Топ топонимов по частоте")
    top = agg.head(15)
    fig = px.bar(top, x="n", y="toponym_name", orientation="h", text="n")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=440, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
with c2:
    st.subheader("Виды туризма (доли)")
    tp = fm[fm["topic_category"] != ""]["topic_category"].value_counts().reset_index()
    tp.columns = ["topic", "cnt"]
    if not tp.empty:
        fig = px.pie(tp, names="topic", values="cnt", hole=0.35)
        fig.update_traces(textposition="inside", textinfo="percent+label", sort=False)
        fig.update_layout(height=440, margin=dict(l=0, r=0, t=10, b=0),
                          legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Под фильтр нет тематики.")

# -------------------- Облако слов по тональности (все слова / фильтр) --------------------
@fragment
def _wordcloud_section(mentions, syntax, fm):
    st.subheader("Облако слов · размер = частота, цвет = тональность")
    st.caption("🟩 позитивный контекст · 🟥 негативный · ⬜ нейтральный")
    kind = st.radio("Тип слов", ["Все", "Топонимы", "Существительные", "Глаголы", "Прилагательные"],
                    horizontal=True, key="wc_kind")
    post_ids = set(fm["post_id"]) if "post_id" in fm.columns else None
    ws = word_stats(mentions, syntax, post_ids)
    _kmap = {"Топонимы": "топоним", "Существительные": "существительное",
             "Глаголы": "глагол", "Прилагательные": "прилагательное"}
    if kind in _kmap:
        ws = ws[ws["kind"] == _kmap[kind]]
    wc_fig = _wordcloud_fig(ws)
    if wc_fig is not None:
        st.pyplot(wc_fig)
    else:
        st.info("Под выбранные фильтры нет слов для облака.")


_wordcloud_section(mentions, syntax, fm)


# -------------------- Топоним → что делать (глаголы) --------------------
@fragment
def _verbs_section(syntax, agg):
    st.subheader("Топоним → что делать (глаголы)")
    verbs_syn = (syntax[syntax["pos"] == "VERB"]
                 if (syntax is not None and not syntax.empty and "pos" in syntax) else pd.DataFrame())
    if verbs_syn.empty:
        st.info("Нет данных по глаголам.")
        return
    places = sorted(agg["toponym_name"].unique())
    default = "Байкал (озеро)" if "Байкал (озеро)" in places else (places[0] if places else None)
    if not default:
        return
    place = st.selectbox("Место", places, index=places.index(default))
    vv = (verbs_syn[verbs_syn["toponym_name"] == place]["normal_form"]
          .value_counts().head(15).reset_index())
    vv.columns = ["verb", "cnt"]
    if vv.empty:
        st.info(f"Для «{place}» нет извлечённых глаголов.")
        return
    fig = px.bar(vv, x="cnt", y="verb", orientation="h", text="cnt")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=340, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")


_verbs_section(syntax, agg)

st.divider()
st.caption("Конвейер: Natasha (NER+синтаксис) + pymorphy3 → геокодинг (газеттир+Nominatim, PostGIS) → "
           "тональность/тематика по топониму. Данные обновляются ежедневным прогоном `analytics.pipeline`.")
