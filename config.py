"""
Конфигурация системы мониторинга медиапространства туризма Иркутской области
"""
import os
from dotenv import load_dotenv
import re
from parsers.sites import (
    IrkRuParser,
    TravelBaikalParser,
    IrCityParser,
    AifIrkParser,
    IrkRaionParser,
    IrkutskoinformParser,
)

load_dotenv()

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ВКонтакте
VK_ACCESS_TOKEN = os.getenv("VK_ACCESS_TOKEN")

# Telegram
TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID")
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")
TELEGRAM_PHONE = os.getenv("TELEGRAM_PHONE")
TELEGRAM_SESSION_STRING = os.getenv("TELEGRAM_SESSION_STRING")

# Telegram Bot (для отправки отчетов)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# Поддержка нескольких chat_id через запятую: "123456,789012,-1001234567890"
_chat_ids_raw = os.getenv("TELEGRAM_CHAT_IDS", "")
TELEGRAM_CHAT_IDS = [cid.strip() for cid in _chat_ids_raw.split(",") if cid.strip()]

# HuggingFace
HF_REPO = os.getenv("HF_REPO")

# Новостные источники
# Конфигурация вынесена в отдельные парсеры: parsers/sites/
NEWS_SOURCES = [
    ("IRK.ru", "https://www.irk.ru", IrkRuParser),
    ("Travel-Baikal.info", "https://travel-baikal.info", TravelBaikalParser),
    ("IrCity.ru", "https://ircity.ru", IrCityParser),
    ("АиФ Иркутск", "https://irk.aif.ru", AifIrkParser),
    ("Иркутский район", "https://www.irkraion.ru", IrkRaionParser),
    ("Иркутскинформ", "https://xn--h1aafalfhlffkls.xn--p1ai", IrkutskoinformParser),
]

# Группы ВКонтакте (ID или screen_name)
VK_GROUPS = [
    "visitirkutskregion",
    "sgt_isu",
    "habara_group",
    "myirkutsk_info",
    "kudap_irkutsk",
    "baikalpro38",
    "baikalburyatia",
    "baikal_retreat_olhon",
]

# Telegram каналы
TELEGRAM_CHANNELS = [
    "Baikal_Daily",
    "Baikal_People",
    "ircity_ru",
    "irkru",
    "admirk",
]

# Настройки парсинга
PARSE_SETTINGS = {
    "max_posts_per_source": 80,
    "days_lookback": 2,
    "min_text_length": 100,
    "min_comment_length": 20,
    "request_delay": 1.0,
}

# Настройки ML классификатора
ML_SETTINGS = {
    # Общие настройки
    "relevance_threshold": 0.5,
    "classifier_type": "bert",  # "tfidf" или "bert"

    # TF-IDF настройки (fallback)
    "tfidf_model_path": "models/classifier.joblib",
    "tfidf_vectorizer_path": "models/vectorizer.joblib",

    # BERT настройки
    "bert_model_path": HF_REPO,
    "bert_base_model": "cointegrated/rubert-tiny2",
    "bert_max_length": 512,
    "bert_batch_size": 16,
    "bert_epochs": 3,
    "bert_learning_rate": 2e-5,
}
