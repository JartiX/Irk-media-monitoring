-- Миграция: добавление поля updated_at в таблицы posts и comments

-- ============================================
-- Добавляем updated_at в таблицу posts
-- ============================================
ALTER TABLE posts
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;

-- Устанавливаем значение для существующих записей
UPDATE posts
SET updated_at = created_at
WHERE updated_at IS NULL;

-- Устанавливаем значение по умолчанию для новых записей
ALTER TABLE posts
ALTER COLUMN updated_at SET DEFAULT NOW();

-- Создаем индекс для поиска по дате обновления
CREATE INDEX IF NOT EXISTS idx_posts_updated ON posts(updated_at DESC);

-- Комментарий
COMMENT ON COLUMN posts.updated_at IS 'Дата последнего обновления записи';

-- ============================================
-- Добавляем updated_at в таблицу comments
-- ============================================
ALTER TABLE comments
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;

-- Устанавливаем значение для существующих записей
UPDATE comments
SET updated_at = created_at
WHERE updated_at IS NULL;

-- Устанавливаем значение по умолчанию для новых записей
ALTER TABLE comments
ALTER COLUMN updated_at SET DEFAULT NOW();

-- Создаем индекс для поиска по дате обновления
CREATE INDEX IF NOT EXISTS idx_comments_updated ON comments(updated_at DESC);

-- Комментарий
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

SELECT 'Миграция updated_at успешно выполнена!' as status;
