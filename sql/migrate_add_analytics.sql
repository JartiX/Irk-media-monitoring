-- ============================================================
-- Миграция: слой аналитики топонимов (практика)
-- Идемпотентна (IF NOT EXISTS / DROP POLICY IF EXISTS) — можно прогонять повторно.
--
-- ЧТО МЕНЯЕМ В СУЩЕСТВУЮЩЕМ:
--   * posts: добавляем 1 опциональную колонку analyzed_at (watermark инкремента).
--     Больше ничего в sources / posts / comments НЕ трогаем.
--
-- ЧТО ДОБАВЛЯЕМ:
--   * 3 таблицы: toponyms, toponym_mentions, toponym_syntax
--   * 4 вьюхи-агрегата: toponym_coords, toponym_tonality, toponym_topics, toponym_verbs
--   * индексы, триггер updated_at, RLS-политики (анон-чтение / service-запись)
--   * расширение PostGIS + пространственная колонка geom (geography Point/4326) + GIST-индекс
-- ============================================================

-- ------------------------------------------------------------
-- PostGIS: пространственное расширение (альтернативно включается через
-- Dashboard -> Database -> Extensions -> postgis). Работаем в ГИС-парадигме.
-- ------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS postgis;

-- ------------------------------------------------------------
-- 0. ИЗМЕНЕНИЕ существующей таблицы posts: watermark инкремента
--    Пайплайн берёт посты, где analyzed_at IS NULL OR analyzed_at < updated_at
--    (новые + изменённые после прошлого прогона), и проставляет analyzed_at = NOW().
--    Колонка nullable — существующий код и записи не затрагиваются.
-- ------------------------------------------------------------
ALTER TABLE posts ADD COLUMN IF NOT EXISTS analyzed_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_posts_analyzed ON posts(analyzed_at);
COMMENT ON COLUMN posts.analyzed_at IS 'Когда пост обработан слоем аналитики топонимов (NULL = ещё не обработан)';

-- ------------------------------------------------------------
-- 1. Справочник топонимов + координаты (аналог Coord_filtr_all)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS toponyms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,              -- нормализованная форма (lower): 'байкал'
    display_name TEXT,                      -- для показа: 'Байкал'
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    -- Пространственная точка (PostGIS), генерируется из lat/lon — основа ГИС-операций.
    -- Пайплайн по-прежнему пишет только lat/lon; geom вычисляется автоматически.
    geom geography(Point, 4326) GENERATED ALWAYS AS (
        ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography
    ) STORED,
    geocode_source TEXT CHECK (geocode_source IN ('gazetteer', 'nominatim', 'manual')),
    gazetteer_id TEXT,                      -- id из Irk_obl_sights.csv, если был матч
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_toponyms_name ON toponyms(name);
CREATE INDEX IF NOT EXISTS idx_toponyms_geom ON toponyms USING GIST (geom);

COMMENT ON TABLE toponyms IS 'Справочник топонимов с координатами (геокодинг: газеттир/Nominatim)';
COMMENT ON COLUMN toponyms.name IS 'Нормализованная форма топонима (ключ дедупликации)';
COMMENT ON COLUMN toponyms.geocode_source IS 'Источник координат: gazetteer | nominatim | manual';

-- ------------------------------------------------------------
-- 2. Упоминания топонимов (аналог Toponims_all): одна строка = топоним в предложении
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS toponym_mentions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    toponym_id UUID NOT NULL REFERENCES toponyms(id) ON DELETE CASCADE,
    toponym_name TEXT NOT NULL,             -- денормализовано (для дашборда без join)
    source_type TEXT,                       -- news/vk/telegram (денормализ. из sources)
    published_at TIMESTAMPTZ,               -- денормализ. из posts (фильтр по времени)
    sentence_idx INTEGER NOT NULL,          -- номер предложения в посте
    sentence TEXT,                          -- текст предложения
    word TEXT,                              -- как встретилось ('Байкала')
    sentence_sentiment REAL DEFAULT 0,      -- тональность предложения (Этап 6)
    topic_category TEXT,                    -- вид туризма (Этап 5)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (post_id, sentence_idx, toponym_id)
);
CREATE INDEX IF NOT EXISTS idx_mentions_toponym ON toponym_mentions(toponym_id);
CREATE INDEX IF NOT EXISTS idx_mentions_post ON toponym_mentions(post_id);
CREATE INDEX IF NOT EXISTS idx_mentions_published ON toponym_mentions(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_mentions_source ON toponym_mentions(source_type);
CREATE INDEX IF NOT EXISTS idx_mentions_topic ON toponym_mentions(topic_category);

COMMENT ON TABLE toponym_mentions IS 'Упоминания топонимов в предложениях постов';
COMMENT ON COLUMN toponym_mentions.sentence_sentiment IS 'Тональность предложения с топонимом (>0 позитив, <0 негатив)';

-- ------------------------------------------------------------
-- 3. Синтаксические связи топоним -> слово/глагол (аналог Syntax_dop)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS toponym_syntax (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    toponym_id UUID NOT NULL REFERENCES toponyms(id) ON DELETE CASCADE,
    toponym_name TEXT NOT NULL,
    word TEXT NOT NULL,                     -- связанное слово как в тексте
    normal_form TEXT,                       -- лемма слова (pymorphy3)
    pos TEXT,                               -- часть речи: VERB / ADJ / NOUN ...
    deprel TEXT,                            -- тип синт. связи: amod, obj, nmod ...
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_syntax_toponym ON toponym_syntax(toponym_id);
CREATE INDEX IF NOT EXISTS idx_syntax_post ON toponym_syntax(post_id);
CREATE INDEX IF NOT EXISTS idx_syntax_pos ON toponym_syntax(pos);

COMMENT ON TABLE toponym_syntax IS 'Слова, синтаксически связанные с топонимом (основа "топоним -> что делать")';

-- ------------------------------------------------------------
-- 4. Триггер updated_at для toponyms (переиспользуем существующую функцию)
-- ------------------------------------------------------------
DROP TRIGGER IF EXISTS trigger_toponyms_updated_at ON toponyms;
CREATE TRIGGER trigger_toponyms_updated_at
    BEFORE UPDATE ON toponyms
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ------------------------------------------------------------
-- 5. Вьюхи-агрегаты (всегда согласованы, пайплайн их НЕ пишет)
-- ------------------------------------------------------------

-- 5.1. Топоним -> координата + частота (аналог Coord_filtr_all)
CREATE OR REPLACE VIEW toponym_coords AS
SELECT
    t.name, t.display_name, t.lat, t.lon,
    ST_AsGeoJSON(t.geom)::json AS geojson,
    COUNT(m.id) AS n_repeat
FROM toponyms t
LEFT JOIN toponym_mentions m ON m.toponym_id = t.id
GROUP BY t.id;

-- 5.2. Тональность по топониму (аналог Toponims_ton)
CREATE OR REPLACE VIEW toponym_tonality AS
SELECT
    toponym_name AS name,
    SUM(GREATEST(sentence_sentiment, 0))   AS positiv,
    SUM(LEAST(sentence_sentiment, 0))      AS negative,
    SUM(sentence_sentiment)                AS sum_ton,
    COUNT(*)                               AS n_mentions
FROM toponym_mentions
GROUP BY toponym_name;

-- 5.3. Тематика: топоним x вид туризма (новое)
CREATE OR REPLACE VIEW toponym_topics AS
SELECT toponym_name, topic_category, COUNT(*) AS cnt
FROM toponym_mentions
WHERE topic_category IS NOT NULL
GROUP BY toponym_name, topic_category;

-- 5.4. Топоним -> топ глаголов ("что делать")
CREATE OR REPLACE VIEW toponym_verbs AS
SELECT toponym_name, normal_form AS verb, COUNT(*) AS cnt
FROM toponym_syntax
WHERE pos = 'VERB'
GROUP BY toponym_name, normal_form;

-- ------------------------------------------------------------
-- 6. RLS: анонимное чтение (для дашборда) + запись только service_role (пайплайн)
--    Зеркалит политики существующих таблиц.
-- ------------------------------------------------------------
ALTER TABLE toponyms          ENABLE ROW LEVEL SECURITY;
ALTER TABLE toponym_mentions  ENABLE ROW LEVEL SECURITY;
ALTER TABLE toponym_syntax    ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow anonymous read access on toponyms" ON toponyms;
CREATE POLICY "Allow anonymous read access on toponyms" ON toponyms FOR SELECT USING (true);
DROP POLICY IF EXISTS "Allow service write on toponyms" ON toponyms;
CREATE POLICY "Allow service write on toponyms" ON toponyms FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "Allow anonymous read access on toponym_mentions" ON toponym_mentions;
CREATE POLICY "Allow anonymous read access on toponym_mentions" ON toponym_mentions FOR SELECT USING (true);
DROP POLICY IF EXISTS "Allow service write on toponym_mentions" ON toponym_mentions;
CREATE POLICY "Allow service write on toponym_mentions" ON toponym_mentions FOR ALL USING (auth.role() = 'service_role');

DROP POLICY IF EXISTS "Allow anonymous read access on toponym_syntax" ON toponym_syntax;
CREATE POLICY "Allow anonymous read access on toponym_syntax" ON toponym_syntax FOR SELECT USING (true);
DROP POLICY IF EXISTS "Allow service write on toponym_syntax" ON toponym_syntax;
CREATE POLICY "Allow service write on toponym_syntax" ON toponym_syntax FOR ALL USING (auth.role() = 'service_role');

-- ------------------------------------------------------------
-- 7. Пример ГИС-запроса (PostGIS): топонимы в радиусе N км от точки.
--    Демонстрирует пространственный поиск; вызывается из дашборда как RPC.
-- ------------------------------------------------------------
CREATE OR REPLACE FUNCTION toponyms_within_km(
    p_lat DOUBLE PRECISION, p_lon DOUBLE PRECISION, p_km DOUBLE PRECISION
)
RETURNS TABLE(name TEXT, lat DOUBLE PRECISION, lon DOUBLE PRECISION, distance_km DOUBLE PRECISION) AS $$
    SELECT t.name, t.lat, t.lon,
           ST_Distance(t.geom, ST_SetSRID(ST_MakePoint(p_lon, p_lat), 4326)::geography) / 1000.0
    FROM toponyms t
    WHERE t.geom IS NOT NULL
      AND ST_DWithin(t.geom, ST_SetSRID(ST_MakePoint(p_lon, p_lat), 4326)::geography, p_km * 1000)
    ORDER BY ST_Distance(t.geom, ST_SetSRID(ST_MakePoint(p_lon, p_lat), 4326)::geography);
$$ LANGUAGE sql STABLE;

SELECT 'Миграция аналитики применена!' AS status;
