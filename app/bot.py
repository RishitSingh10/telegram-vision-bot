"""
Telegram Vision Bot - Main Entry Point
Handles all Telegram interactions and routes to handlers.
"""

import logging
import asyncio
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from config import Config
from handlers import (
    start_handler,
    help_handler,
    image_handler,
    summarize_handler,
    text_message_handler,
    photo_handler,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def create_application() -> Application:
    """Create and configure the Telegram application."""
    config = Config()
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    # Command handlers
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("image", image_handler))
    app.add_handler(CommandHandler("summarize", summarize_handler))

    # Message handlers
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))

    logger.info("Bot application configured successfully.")
    return app


def main():
    """Start the bot."""
    logger.info("Starting Telegram Vision Bot...")
    app = create_application()
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()