"""
Configuration management using environment variables.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Telegram
    TELEGRAM_BOT_TOKEN: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", "")
    )

    # Vision Model
    VISION_MODEL: str = field(
        default_factory=lambda: os.getenv("VISION_MODEL", "Salesforce/blip2-opt-2.7b")
    )
    # Options: "Salesforce/blip2-opt-2.7b", "llava-hf/llava-1.5-7b-hf",
    #          "nlpconnect/vit-gpt2-image-captioning" (lightweight)

    # Use lightweight model by default for easier local setup
    USE_LIGHTWEIGHT_MODEL: bool = field(
        default_factory=lambda: os.getenv("USE_LIGHTWEIGHT_MODEL", "true").lower() == "true"
    )

    # History & Cache
    MAX_HISTORY_PER_USER: int = 3
    CACHE_MAX_SIZE: int = 128  # Max cached image hashes

    # Gradio UI
    GRADIO_PORT: int = field(
        default_factory=lambda: int(os.getenv("GRADIO_PORT", "7860"))
    )
    GRADIO_HOST: str = field(
        default_factory=lambda: os.getenv("GRADIO_HOST", "0.0.0.0")
    )

    # Logging
    LOG_LEVEL: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    def validate(self):
        if not self.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required.")
        return self