"""
Session Manager
Maintains per-user message history (last 3 interactions).
Stores image bytes and captions for /summarize functionality.
"""

import logging
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

MAX_HISTORY = 3


@dataclass
class Interaction:
    """A single user interaction (text or image)."""
    type: str          # "text" | "image"
    user_input: str    # user's message or image filename
    bot_response: str  # bot's reply
    caption: Optional[str] = None   # for image interactions
    tags: Optional[list] = None     # for image interactions


@dataclass
class UserSession:
    """Session state for a single Telegram user."""
    user_id: int
    history: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY))
    last_image_bytes: Optional[bytes] = None
    last_caption: Optional[str] = None
    last_tags: Optional[list] = None
    awaiting_image: bool = False  # True after /image command

    def add_interaction(self, interaction: Interaction):
        self.history.append(interaction)

    def get_history_text(self) -> str:
        """Format history as readable text for /summarize."""
        if not self.history:
            return "No conversation history yet."

        lines = []
        for i, item in enumerate(self.history, 1):
            lines.append(f"[{i}] {item.type.upper()}")
            lines.append(f"  You: {item.user_input}")
            lines.append(f"  Bot: {item.bot_response}")
        return "\n".join(lines)

    def get_last_image_summary(self) -> str:
        """Return a summary of the last analyzed image."""
        if not self.last_caption:
            return "No image has been analyzed yet."
        tags_str = ", ".join(self.last_tags) if self.last_tags else "none"
        return (
            f"📷 Last image analysis:\n"
            f"Caption: {self.last_caption}\n"
            f"Tags: #{' #'.join(self.last_tags or [])}"
        )


class SessionManager:
    """Thread-safe in-memory session store for all users."""

    def __init__(self):
        self._sessions: dict[int, UserSession] = {}

    def get_session(self, user_id: int) -> UserSession:
        """Get or create a session for a user."""
        if user_id not in self._sessions:
            self._sessions[user_id] = UserSession(user_id=user_id)
            logger.info(f"New session created for user {user_id}")
        return self._sessions[user_id]

    def record_interaction(
        self,
        user_id: int,
        interaction_type: str,
        user_input: str,
        bot_response: str,
        caption: Optional[str] = None,
        tags: Optional[list] = None,
    ):
        session = self.get_session(user_id)
        interaction = Interaction(
            type=interaction_type,
            user_input=user_input,
            bot_response=bot_response,
            caption=caption,
            tags=tags,
        )
        session.add_interaction(interaction)

        if caption:
            session.last_caption = caption
        if tags:
            session.last_tags = tags

    def set_awaiting_image(self, user_id: int, value: bool):
        self.get_session(user_id).awaiting_image = value

    def is_awaiting_image(self, user_id: int) -> bool:
        return self.get_session(user_id).awaiting_image

    def store_last_image(self, user_id: int, image_bytes: bytes):
        self.get_session(user_id).last_image_bytes = image_bytes

    def get_all_stats(self) -> dict:
        return {
            "total_users": len(self._sessions),
            "users_with_history": sum(
                1 for s in self._sessions.values() if len(s.history) > 0
            ),
        }


# Global singleton
session_manager = SessionManager()