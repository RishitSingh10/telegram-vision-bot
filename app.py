"""
app.py — Unified entry point.
Starts the Telegram bot and optionally the Gradio debug UI.

Usage:
    python app.py              # Bot only
    python app.py --ui         # Bot + Gradio UI
    python app.py --ui-only    # Gradio UI only (no Telegram)
"""

import argparse
import logging
import threading
import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def run_gradio():
    """Start Gradio UI in a separate thread."""
    from app.gradio_ui import demo
    from app.config import Config
    config = Config()
    logger.info(f"Starting Gradio UI on http://{config.GRADIO_HOST}:{config.GRADIO_PORT}")
    demo.launch(
        server_name=config.GRADIO_HOST,
        server_port=config.GRADIO_PORT,
        share=False,
        prevent_thread_lock=True,
    )


def run_bot():
    """Start the Telegram bot."""
    from app.bot import main as bot_main
    from app.config import Config
    config = Config()
    try:
        config.validate()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    logger.info("Starting Telegram bot...")
    bot_main()


def main():
    parser = argparse.ArgumentParser(description="Telegram Vision Bot")
    parser.add_argument("--ui", action="store_true", help="Also start Gradio UI")
    parser.add_argument("--ui-only", action="store_true", help="Start only Gradio UI")
    args = parser.parse_args()

    if args.ui_only:
        run_gradio()
        # Block main thread
        import time
        while True:
            time.sleep(1)
    elif args.ui:
        # Start Gradio in background thread
        ui_thread = threading.Thread(target=run_gradio, daemon=True)
        ui_thread.start()
        run_bot()
    else:
        run_bot()


if __name__ == "__main__":
    main()