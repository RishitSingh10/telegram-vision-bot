"""
Telegram Bot Handlers
All command and message handlers for the Telegram bot.
"""

import logging
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode, ChatAction

from session_manager import session_manager
from vision_service import VisionService
from config import Config

logger = logging.getLogger(__name__)

# Lazy-load the vision service (heavy model load)
_vision_service: VisionService | None = None


def get_vision_service() -> VisionService:
    global _vision_service
    if _vision_service is None:
        config = Config()
        _vision_service = VisionService(
            use_lightweight=config.USE_LIGHTWEIGHT_MODEL,
            model_name=config.VISION_MODEL if not config.USE_LIGHTWEIGHT_MODEL else None,
        )
    return _vision_service


# ──────────────────────────────────────────────────────────────────────────────
# COMMAND: /start
# ──────────────────────────────────────────────────────────────────────────────

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    text = (
        f"👋 Hello, *{user.first_name}*\\!\n\n"
        "I'm a *Vision Bot* that can analyze and describe images\\.\n\n"
        "📸 Send me an image and I'll describe it\\!\n"
        "Type /help to see all commands\\."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


# ──────────────────────────────────────────────────────────────────────────────
# COMMAND: /help
# ──────────────────────────────────────────────────────────────────────────────

async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "🤖 *Vision Bot — Help*\n\n"
        "*Commands:*\n"
        "• /start — Welcome message\n"
        "• /image — Upload an image for AI description\n"
        "• /summarize — Summarize your last image or chat\n"
        "• /help — Show this help message\n\n"
        "*How to use:*\n"
        "1\\. Send any image directly to the chat\n"
        "2\\. Or type /image, then upload the image\n"
        "3\\. You can also ask a question about the image in the caption\\!\n\n"
        "*What you get back:*\n"
        "• 📝 A short caption describing the image\n"
        "• 🏷 3 keyword tags\n"
        "• ⚡ Cached results for repeated images"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN_V2)


# ──────────────────────────────────────────────────────────────────────────────
# COMMAND: /image
# ──────────────────────────────────────────────────────────────────────────────

async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session_manager.set_awaiting_image(user_id, True)
    await update.message.reply_text(
        "📸 Please send me an image\\! You can also include a question in the image caption\\.",
        parse_mode=ParseMode.MARKDOWN_V2,
    )


# ──────────────────────────────────────────────────────────────────────────────
# COMMAND: /summarize
# ──────────────────────────────────────────────────────────────────────────────

async def summarize_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = session_manager.get_session(user_id)

    parts = []

    # Last image summary
    img_summary = session.get_last_image_summary()
    parts.append(f"*Last Image Analysis:*\n{img_summary}")

    # Conversation history
    history = session.get_history_text()
    parts.append(f"\n*Recent Conversation \\(last {3} interactions\\):*\n```\n{history}\n```")

    response = "\n".join(parts)

    # Telegram has a 4096 char limit
    if len(response) > 4000:
        response = response[:4000] + "\n\\.\\.\\."

    await update.message.reply_text(
        response,
        parse_mode=ParseMode.MARKDOWN_V2,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MESSAGE: Photo received
# ──────────────────────────────────────────────────────────────────────────────

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session_manager.set_awaiting_image(user_id, False)

    # Show "typing..." action
    await update.message.chat.send_action(ChatAction.TYPING)

    # Download the largest photo
    photo = update.message.photo[-1]
    photo_file = await context.bot.get_file(photo.file_id)
    image_bytes = await photo_file.download_as_bytearray()
    image_bytes = bytes(image_bytes)

    # Optional question from caption
    question = update.message.caption if update.message.caption else None

    # Store for /summarize
    session_manager.store_last_image(user_id, image_bytes)

    await update.message.reply_text("🔍 Analyzing your image, please wait...")

    try:
        vision = get_vision_service()
        result = vision.describe_image(image_bytes, question=question)

        caption = result["caption"]
        tags = result["tags"]
        cached = result["cached"]

        tags_str = "  ".join(f"#{t}" for t in tags)
        cache_note = " ⚡ \\(cached\\)" if cached else ""

        response = (
            f"📝 *Caption:*\n{escape_md(caption)}\n\n"
            f"🏷 *Tags:*\n{escape_md(tags_str)}"
            f"{cache_note}"
        )

        if question:
            response = f"❓ *Your question:* {escape_md(question)}\n\n" + response

        await update.message.reply_text(response, parse_mode=ParseMode.MARKDOWN_V2)

        # Log interaction
        session_manager.record_interaction(
            user_id=user_id,
            interaction_type="image",
            user_input=question or "[image uploaded]",
            bot_response=f"Caption: {caption} | Tags: {', '.join(tags)}",
            caption=caption,
            tags=tags,
        )

    except Exception as e:
        logger.error(f"Error analyzing image for user {user_id}: {e}")
        await update.message.reply_text(
            "❌ Sorry, I encountered an error analyzing your image\\. "
            "Please try again or use a different image\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )


# ──────────────────────────────────────────────────────────────────────────────
# MESSAGE: Text message
# ──────────────────────────────────────────────────────────────────────────────

async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text

    if session_manager.is_awaiting_image(user_id):
        await update.message.reply_text(
            "📸 Please send an *image* \\(not text\\)\\. "
            "Use the attachment icon to upload a photo\\.",
            parse_mode=ParseMode.MARKDOWN_V2,
        )
        return

    # General text response
    session = session_manager.get_session(user_id)

    # Context-aware reply based on history
    if session.last_caption:
        reply = (
            f"💬 You said: _{escape_md(text)}_\n\n"
            f"🖼 Your last image was about: _{escape_md(session.last_caption)}_\n\n"
            "Send me a new image anytime, or use /help for more options\\!"
        )
    else:
        reply = (
            f"💬 I received your message\\!\n\n"
            "I'm a *Vision Bot* — send me an image to get started\\!\n"
            "Type /help for instructions\\."
        )

    await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN_V2)

    session_manager.record_interaction(
        user_id=user_id,
        interaction_type="text",
        user_input=text,
        bot_response=reply,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def escape_md(text: str) -> str:
    """Escape special MarkdownV2 characters."""
    special = r"\_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{c}" if c in special else c for c in str(text))