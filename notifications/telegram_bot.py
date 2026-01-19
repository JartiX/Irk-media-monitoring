"""
Отправка отчетов в Telegram бота
"""
import asyncio
import aiohttp
from loguru import logger

import config


async def send_report(stats: dict, elapsed_seconds: float, db_stats: dict | None = None) -> int:
    """
    Отправить отчет о выполнении мониторинга в Telegram

    Args:
        stats: Статистика выполнения (posts_processed, posts_new, etc.)
        elapsed_seconds: Время выполнения в секундах
        db_stats: Статистика из БД (опционально)

    Returns:
        Количество успешно отправленных сообщений
    """
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_IDS:
        logger.warning("TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_IDS не установлены, отчет не отправлен")
        return 0

    # Формируем текст отчета
    message = _format_report(stats, elapsed_seconds, db_stats)

    # Отправляем всем получателям параллельно
    tasks = [_send_telegram_message(chat_id, message) for chat_id in config.TELEGRAM_CHAT_IDS]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success_count = sum(1 for r in results if r is True)
    total = len(config.TELEGRAM_CHAT_IDS)

    if success_count == total:
        logger.info(f"Отчет отправлен всем получателям ({success_count})")
    elif success_count > 0:
        logger.warning(f"Отчет отправлен частично: {success_count}/{total}")
    else:
        logger.error("Не удалось отправить отчет ни одному получателю")

    return success_count


def _format_report(stats: dict, elapsed_seconds: float, db_stats: dict | None) -> str:
    """Форматировать отчет для отправки"""
    lines = [
        "📊 <b>Отчет мониторинга медиа</b>",
        "",
        f"⏱ Время выполнения: {elapsed_seconds:.1f} сек",
        "",
        "📝 <b>Посты:</b>",
        f"   • Обработано: {stats['posts_processed']}",
        f"   • Новых: {stats['posts_new']}",
        f"   • Обновлено: {stats['posts_updated']}",
        f"   • Релевантных: {stats['posts_relevant']}",
        "",
        "💬 <b>Комментарии:</b>",
        f"   • Обработано: {stats['comments_processed']}",
        f"   • Новых: {stats['comments_new']}",
        f"   • Обновлено: {stats['comments_updated']}",
        f"   • Чистых: {stats['comments_clean']}",
        f"   • Релевантных: {stats['comments_relevant']}",
        f"   • С политикой: {stats['comments_political']}",
        f"   • С матом: {stats['comments_profane']}",
    ]

    if stats.get("errors", 0) > 0:
        lines.append("")
        lines.append(f"⚠️ Ошибок: {stats['errors']}")

    if db_stats:
        lines.extend([
            "",
            "📈 <b>База данных:</b>",
            f"   • Источников: {db_stats['sources_count']}",
            f"   • Всего постов: {db_stats['posts_count']}",
            f"   • Релевантных: {db_stats['relevant_posts_count']}",
            f"   • Комментариев: {db_stats['comments_count']}",
        ])

    return "\n".join(lines)


async def _send_telegram_message(chat_id: str, text: str) -> bool:
    """Отправить сообщение в конкретный чат через Telegram Bot API"""
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=30) as response:
                if response.status == 200:
                    logger.debug(f"Отчет отправлен в чат {chat_id}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Ошибка отправки в чат {chat_id}: {response.status} - {error_text}")
                    return False
    except Exception as e:
        logger.error(f"Ошибка отправки в чат {chat_id}: {e}")
        return False
