-- SQL скрипт для создания таблиц в Supabase

-- Включаем расширение для UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- Таблица источников данных
-- ============================================
CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('news', 'vk', 'telegram')),
    url TEXT NOT NULL UNIQUE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Индексы для sources
CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(type);
CREATE INDEX IF NOT EXISTS idx_sources_active ON sources(is_active);

-- Комментарии
COMMENT ON TABLE sources IS 'Источники данных для мониторинга';
COMMENT ON COLUMN sources.type IS 'Тип источника: news, vk, telegram';

-- ============================================
-- Таблица постов/новостей
-- ============================================
CREATE TABLE IF NOT EXISTS posts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    external_id TEXT NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    url TEXT NOT NULL,
    published_at TIMESTAMPTZ,
    likes_count INTEGER DEFAULT 0,
    views_count INTEGER DEFAULT 0,
    comments_count INTEGER DEFAULT 0,
    relevance_score FLOAT DEFAULT 0,
    is_relevant BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Уникальность по источнику и внешнему ID
    UNIQUE(source_id, external_id)
);

-- Индексы для posts
CREATE INDEX IF NOT EXISTS idx_posts_source ON posts(source_id);
CREATE INDEX IF NOT EXISTS idx_posts_relevant ON posts(is_relevant);
CREATE INDEX IF NOT EXISTS idx_posts_published ON posts(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_posts_updated ON posts(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_posts_relevance_score ON posts(relevance_score DESC);

-- Полнотекстовый поиск
ALTER TABLE posts ADD COLUMN IF NOT EXISTS fts tsvector
    GENERATED ALWAYS AS (to_tsvector('russian', coalesce(title, '') || ' ' || content)) STORED;
CREATE INDEX IF NOT EXISTS idx_posts_fts ON posts USING GIN(fts);

-- Комментарии
COMMENT ON TABLE posts IS 'Посты и новости из источников';
COMMENT ON COLUMN posts.external_id IS 'ID поста в исходной системе';
COMMENT ON COLUMN posts.relevance_score IS 'Оценка релевантности (0-1)';
COMMENT ON COLUMN posts.updated_at IS 'Дата последнего обновления записи';

-- ============================================
-- Таблица комментариев
-- ============================================
CREATE TABLE IF NOT EXISTS comments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    external_id TEXT NOT NULL,
    author TEXT,
    content TEXT NOT NULL,
    published_at TIMESTAMPTZ,
    likes_count INTEGER DEFAULT 0,
    is_clean BOOLEAN DEFAULT FALSE,
    is_relevant BOOLEAN DEFAULT FALSE,
    is_political BOOLEAN DEFAULT FALSE,
    is_profane BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Уникальность по посту и внешнему ID
    UNIQUE(post_id, external_id)
);

-- Индексы для comments
CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_id);
CREATE INDEX IF NOT EXISTS idx_comments_clean ON comments(is_clean);
CREATE INDEX IF NOT EXISTS idx_comments_relevant ON comments(is_relevant);
CREATE INDEX IF NOT EXISTS idx_comments_political ON comments(is_political);
CREATE INDEX IF NOT EXISTS idx_comments_profane ON comments(is_profane);
CREATE INDEX IF NOT EXISTS idx_comments_published ON comments(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_comments_updated ON comments(updated_at DESC);

-- Комментарии
COMMENT ON TABLE comments IS 'Комментарии к постам';
COMMENT ON COLUMN comments.is_clean IS 'Чистый комментарий (без политики и мата)';
COMMENT ON COLUMN comments.is_relevant IS 'Комментарий связан с туризмом';
COMMENT ON COLUMN comments.is_political IS 'Комментарий содержит политику';
COMMENT ON COLUMN comments.is_profane IS 'Комментарий содержит нецензурную лексику';
COMMENT ON COLUMN comments.updated_at IS 'Дата последнего обновления записи';

-- ============================================
-- Триггер для автоматического обновления updated_at
-- ============================================

-- Функция для обновления updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Триггер для posts
DROP TRIGGER IF EXISTS trigger_posts_updated_at ON posts;
CREATE TRIGGER trigger_posts_updated_at
    BEFORE UPDATE ON posts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Триггер для comments
DROP TRIGGER IF EXISTS trigger_comments_updated_at ON comments;
CREATE TRIGGER trigger_comments_updated_at
    BEFORE UPDATE ON comments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- Представления для аналитики
-- ============================================

-- Статистика по источникам
CREATE OR REPLACE VIEW source_stats AS
SELECT
    s.id,
    s.name,
    s.type,
    COUNT(p.id) as total_posts,
    COUNT(p.id) FILTER (WHERE p.is_relevant) as relevant_posts,
    AVG(p.relevance_score) FILTER (WHERE p.relevance_score > 0) as avg_relevance,
    MAX(p.published_at) as last_post_date
FROM sources s
LEFT JOIN posts p ON p.source_id = s.id
GROUP BY s.id, s.name, s.type;

-- Последние релевантные посты
CREATE OR REPLACE VIEW recent_relevant_posts AS
SELECT
    p.id,
    s.name as source_name,
    s.type as source_type,
    p.title,
    LEFT(p.content, 200) as content_preview,
    p.url,
    p.published_at,
    p.relevance_score,
    p.likes_count,
    p.views_count
FROM posts p
JOIN sources s ON s.id = p.source_id
WHERE p.is_relevant = TRUE
ORDER BY p.published_at DESC
LIMIT 100;

-- Чистые релевантные комментарии
CREATE OR REPLACE VIEW clean_relevant_comments_view AS
SELECT
    c.id,
    c.author,
    c.content,
    c.published_at,
    c.likes_count,
    c.is_clean,
    c.is_relevant,
    c.is_political,
    c.is_profane,
    p.title as post_title,
    p.url as post_url,
    s.name as source_name
FROM comments c
JOIN posts p ON p.id = c.post_id
JOIN sources s ON s.id = p.source_id
WHERE c.is_clean = TRUE AND c.is_relevant = TRUE
ORDER BY c.published_at DESC;

-- ============================================
-- Row Level Security (RLS)
-- ============================================

-- Включаем RLS для всех таблиц
ALTER TABLE sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE posts ENABLE ROW LEVEL SECURITY;
ALTER TABLE comments ENABLE ROW LEVEL SECURITY;

-- Политики для чтения (разрешаем всем анонимным пользователям)
CREATE POLICY "Allow anonymous read access on sources" ON sources
    FOR SELECT USING (true);

CREATE POLICY "Allow anonymous read access on posts" ON posts
    FOR SELECT USING (true);

CREATE POLICY "Allow anonymous read access on comments" ON comments
    FOR SELECT USING (true);

-- Политики для записи (только с сервисным ключом)
CREATE POLICY "Allow service write on sources" ON sources
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Allow service write on posts" ON posts
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Allow service write on comments" ON comments
    FOR ALL USING (auth.role() = 'service_role');

-- ============================================
-- Функции для статистики
-- ============================================

-- Функция получения общей статистики
CREATE OR REPLACE FUNCTION get_monitoring_stats()
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'sources_count', (SELECT COUNT(*) FROM sources WHERE is_active = TRUE),
        'posts_count', (SELECT COUNT(*) FROM posts),
        'relevant_posts_count', (SELECT COUNT(*) FROM posts WHERE is_relevant = TRUE),
        'comments_count', (SELECT COUNT(*) FROM comments),
        'clean_comments_count', (SELECT COUNT(*) FROM comments WHERE is_clean = TRUE),
        'relevant_comments_count', (SELECT COUNT(*) FROM comments WHERE is_relevant = TRUE),
        'political_comments_count', (SELECT COUNT(*) FROM comments WHERE is_political = TRUE),
        'profane_comments_count', (SELECT COUNT(*) FROM comments WHERE is_profane = TRUE),
        'last_update', (SELECT MAX(updated_at) FROM posts)
    ) INTO result;

    RETURN result;
END;
$$ LANGUAGE plpgsql;

SELECT 'Таблицы успешно созданы!' as status;
