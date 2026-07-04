-- Миграция: тональность предложения в связях «топоним → слово».
-- Нужна для облака слов на дашборде с цветовой кодировкой тональности (глаголы/прилагательные
-- окрашиваются по тональности предложений, в которых они встречаются).
--
-- Применение: Supabase → SQL Editor → выполнить. Затем перезаписать данные:
--   .venv\Scripts\python.exe -m analytics.pipeline --source supabase --to-db

ALTER TABLE toponym_syntax ADD COLUMN IF NOT EXISTS sentence_sentiment REAL DEFAULT 0;

COMMENT ON COLUMN toponym_syntax.sentence_sentiment IS
    'Тональность предложения, где найдена связь (для окраски слов в облаке: >0 позитив, <0 негатив)';
