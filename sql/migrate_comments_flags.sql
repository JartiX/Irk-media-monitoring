-- Миграция таблицы comments: замена is_useful на 4 новых флага

-- 1. Удаляем старое представление
DROP VIEW IF EXISTS useful_comments_view;

-- 2. Добавляем новые колонки
ALTER TABLE comments ADD COLUMN IF NOT EXISTS is_clean BOOLEAN DEFAULT FALSE;
ALTER TABLE comments ADD COLUMN IF NOT EXISTS is_relevant BOOLEAN DEFAULT FALSE;
ALTER TABLE comments ADD COLUMN IF NOT EXISTS is_political BOOLEAN DEFAULT FALSE;
ALTER TABLE comments ADD COLUMN IF NOT EXISTS is_profane BOOLEAN DEFAULT FALSE;

-- 3. Миграция данных: is_useful = TRUE -> is_clean = TRUE, is_relevant = TRUE
UPDATE comments SET is_clean = TRUE, is_relevant = TRUE WHERE is_useful = TRUE;
UPDATE comments SET is_clean = TRUE WHERE is_useful = FALSE OR is_useful IS NULL;

-- 4. Удаляем старую колонку
ALTER TABLE comments DROP COLUMN IF EXISTS is_useful;

-- 5. Создаём индексы
CREATE INDEX IF NOT EXISTS idx_comments_clean ON comments(is_clean);
CREATE INDEX IF NOT EXISTS idx_comments_relevant ON comments(is_relevant);
CREATE INDEX IF NOT EXISTS idx_comments_political ON comments(is_political);
CREATE INDEX IF NOT EXISTS idx_comments_profane ON comments(is_profane);

-- 6. Удаляем старый индекс
DROP INDEX IF EXISTS idx_comments_useful;

-- 7. Добавляем комментарии
COMMENT ON COLUMN comments.is_clean IS 'Чистый комментарий (без политики и мата)';
COMMENT ON COLUMN comments.is_relevant IS 'Комментарий связан с туризмом';
COMMENT ON COLUMN comments.is_political IS 'Комментарий содержит политику';
COMMENT ON COLUMN comments.is_profane IS 'Комментарий содержит нецензурную лексику';

-- 8. Обновляем представление

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

-- 9. Обновляем функцию статистики
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
        'last_update', (SELECT MAX(created_at) FROM posts)
    ) INTO result;

    RETURN result;
END;
$$ LANGUAGE plpgsql;

SELECT 'Миграция завершена!' as status;
