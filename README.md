# Irk Media Monitoring

Автоматизированная система мониторинга медиапространства туризма Иркутской области. Собирает и фильтрует контент о туризме в Иркутской области с новостных сайтов, групп ВКонтакте и Telegram-каналов.

## Возможности

- Парсинг новостных сайтов (IRK.ru, IrCity.ru, Travel-Baikal.info и др.)
- Парсинг групп ВКонтакте через VK API
- Парсинг Telegram-каналов через Telethon
- Двухэтапная фильтрация: ключевые слова + ML-классификатор (BERT/TF-IDF)
- Хранение данных в Supabase
- Ежедневный автоматический запуск через GitHub Actions
- Отправка отчетов в Telegram-бот

## Quick Start

### 1. Клонирование репозитория

```bash
git clone https://github.com/your-username/Irk-media-monitoring.git
cd Irk-media-monitoring
```

### 2. Создание виртуального окружения

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Установка зависимостей

**Базовая установка (CPU, без BERT):**
```bash
pip install -r requirements.txt
```

**Установка с BERT(NO GPU):**
```bash
pip install -r requirements.txt
pip install -r requirements_bert.txt
```

**Дополнительно:**

Для ускорения ML-классификации на GPU установите PyTorch с CUDA. Выберите версию, соответствующую вашей видеокарте:

```bash
# CUDA 12.6 (рекомендуется для новых GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

> Актуальные команды установки: https://pytorch.org/get-started/locally/

### 4. Настройка переменных окружения

Скопируйте `.env.example` в `.env` и заполните необходимые значения:

```bash
cp .env.example .env
```

**Обязательные переменные:**
- `SUPABASE_URL`, `SUPABASE_KEY` — подключение к базе данных

**Опциональные:**
- `VK_ACCESS_TOKEN` — для парсинга ВКонтакте
- `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_SESSION_STRING`, `TELEGRAM_PHONE` — для парсинга Telegram
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_IDS` — для отправки отчетов
- `HF_REPO` — для сохранения / загрузки BERT модели

### 5. Создание таблиц в Supabase

Выполните SQL-скрипт в Supabase Dashboard:

```sql
-- Содержимое sql/create_tables.sql
```

### 6. Запуск

```bash
# Windows
.venv\Scripts\python.exe main.py

# Linux/macOS
python main.py
```

## Архитектура

```
Sources → Parsers → Filters → Supabase
```

### Структура проекта

```
Irk-media-monitoring/
├── main.py                 # Точка входа
├── config.py               # Конфигурация (API-ключи, источники, ключевые слова)
├── requirements.txt        # Зависимости
├── .env.example           # Пример переменных окружения
│
├── parsers/               # Парсеры источников
│   ├── base.py            # Абстрактный базовый класс
│   ├── base_news.py       # Базовый класс для новостных сайтов
│   ├── vk_parser.py       # Парсер ВКонтакте
│   ├── telegram_parser.py # Парсер Telegram
│   └── sites/             # Парсеры конкретных сайтов
│       ├── irk_ru.py
│       ├── ircity.py
│       ├── travel_baikal.py
│       ├── aif_irk.py
│       ├── irkraion.py
│       └── irkutskoinform.py
│
├── filters/               # Фильтрация контента
│   ├── keywords.py        # Фильтр по ключевым словам
│   ├── ml_classifier.py   # ML-классификатор (обертка)
│   ├── bert_classifier.py # BERT-классификатор
│   └── tfidf_classifier.py # TF-IDF классификатор
│
├── database/              # Работа с БД
│   ├── models.py          # Модели данных (Post, Source, Comment)
│   └── supabase_client.py # Клиент Supabase
│
├── notifications/         # Уведомления
│   └── telegram_bot.py    # Отправка отчетов в Telegram
│
├── scripts/               # Вспомогательные скрипты
│   └── train_bert.py      # Обучение BERT-классификатора
│
├── sql/                   # SQL-скрипты
│   └── create_tables.sql  # Создание таблиц в Supabase
│
└── .github/workflows/     # GitHub Actions
    └── daily_parse.yml    # Ежедневный запуск
```

## ML-классификация

Система использует двухэтапную фильтрацию:

1. **KeywordFilter** — быстрая фильтрация по ключевым словам
2. **MLClassifier** — точная классификация с помощью ML-модели

### Доступные классификаторы

| Тип | Модель | Качество | Размер | Скорость |
|-----|--------|----------|--------|----------|
| `bert` | rubert-tiny2 (fine-tuned) | Высокое | ~110MB | ~50ms/пост (CPU) |
| `tfidf` | TF-IDF + LogisticRegression | Базовое | ~5MB | <1ms/пост |

### Переключение классификатора

В `config.py` измените `ML_SETTINGS["classifier_type"]`:

```python
ML_SETTINGS = {
    "classifier_type": "bert",  # или "tfidf"
    # ...
}
```

### Обучение BERT-модели

```bash
# Базовое обучение
.venv\Scripts\python.exe scripts/train_bert.py

# С кастомными параметрами
.venv\Scripts\python.exe scripts/train_bert.py --epochs 5 --batch-size 8
```

## Источники данных

### Новостные сайты

| Источник | URL |
|----------|-----|
| IRK.ru | https://www.irk.ru/tourism/ |
| IrCity.ru | https://ircity.ru/text/tags/turizm/ |
| Travel-Baikal.info | https://travel-baikal.info/news |
| АиФ Иркутск | https://irk.aif.ru/tag/turizm |
| Иркутский район | https://www.irkraion.ru/news/turizm |
| Иркутскинформ | https://приангарье.рф/news/category/turizm/ |

### Telegram-каналы

- `@Baikal_Daily`
- `@Baikal_People`
- `@ircity_ru`
- `@irkru`
- `@admirk`

### ВКонтакте

- `visitirkutskregion`
- `sgt_isu`
- `habara_group`
- `myirkutsk_info`
- `kudap_irkutsk`
- `baikalpro38"`
- `baikalburyatia`
- `baikal_retreat_olhon`

## Получение API-ключей

### Supabase

1. Создайте проект на https://supabase.com
2. Перейдите в Settings → API
3. Скопируйте `URL` и `anon key`

### ВКонтакте

1. Создайте приложение на https://vk.com/dev
2. Тип приложения: Standalone
3. Скопируйте сервисный ключ доступа

### Telegram (для парсинга)

1. Перейдите на https://my.telegram.org
2. Создайте приложение в "API development tools"
3. Скопируйте `api_id` и `api_hash`
4. Сгенерируйте session string:

```bash
.venv\Scripts\python.exe -c "import asyncio; from parsers.telegram_parser import TelegramParser; asyncio.run(TelegramParser.generate_session_string())"
```

### Telegram Bot (для отчетов)

1. Создайте бота через @BotFather
2. Скопируйте токен бота
3. Получите chat_id через https://api.telegram.org/bot<TOKEN>/getUpdates

## GitHub Actions

Workflow `.github/workflows/daily_parse.yml` запускается ежедневно в 06:00 UTC.

### Настройка секретов

В Settings → Secrets and variables → Actions добавьте:
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `VK_ACCESS_TOKEN`
- `TELEGRAM_API_ID`
- `TELEGRAM_API_HASH`
- `TELEGRAM_SESSION_STRING`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_IDS`
- `TELEGRAM_PHONE`
- `HF_REPO` (для хранения модели на HuggingFace)
- `HF_TOKEN` (для авторизации в HuggingFace)

## Лицензия

MIT
