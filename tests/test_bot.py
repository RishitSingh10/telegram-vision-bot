"""
Tests for Vision Bot components.
Run with: pytest tests/
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from io import BytesIO
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def make_test_image(width=100, height=100, color=(255, 0, 0)) -> bytes:
    """Create a simple red test image."""
    img = Image.new("RGB", (width, height), color=color)
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


# ──────────────────────────────────────────────────────────────────────────────
# Session Manager Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSessionManager:
    def setup_method(self):
        """Fresh session manager for each test."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
        from session_manager import SessionManager
        self.sm = SessionManager()

    def test_creates_new_session(self):
        session = self.sm.get_session(user_id=1)
        assert session.user_id == 1
        assert len(session.history) == 0

    def test_same_user_same_session(self):
        s1 = self.sm.get_session(1)
        s2 = self.sm.get_session(1)
        assert s1 is s2

    def test_history_limit(self):
        """History should be capped at 3 (MAX_HISTORY)."""
        for i in range(5):
            self.sm.record_interaction(
                user_id=1,
                interaction_type="text",
                user_input=f"msg {i}",
                bot_response=f"reply {i}",
            )
        session = self.sm.get_session(1)
        assert len(session.history) == 3

    def test_last_interaction_in_history(self):
        for i in range(5):
            self.sm.record_interaction(
                user_id=2,
                interaction_type="text",
                user_input=f"msg {i}",
                bot_response=f"reply {i}",
            )
        session = self.sm.get_session(2)
        # The last item should be msg 4
        assert "msg 4" in session.history[-1].user_input

    def test_awaiting_image_flag(self):
        self.sm.set_awaiting_image(3, True)
        assert self.sm.is_awaiting_image(3) is True
        self.sm.set_awaiting_image(3, False)
        assert self.sm.is_awaiting_image(3) is False

    def test_store_last_image(self):
        image_bytes = make_test_image()
        self.sm.store_last_image(4, image_bytes)
        session = self.sm.get_session(4)
        assert session.last_image_bytes == image_bytes

    def test_get_history_text_empty(self):
        session = self.sm.get_session(99)
        text = session.get_history_text()
        assert "No conversation history" in text

    def test_get_last_image_summary_empty(self):
        session = self.sm.get_session(100)
        summary = session.get_last_image_summary()
        assert "No image" in summary

    def test_get_last_image_summary_with_data(self):
        self.sm.record_interaction(
            user_id=5,
            interaction_type="image",
            user_input="[image]",
            bot_response="A cat sitting on a mat",
            caption="A cat sitting on a mat",
            tags=["cat", "mat", "sitting"],
        )
        session = self.sm.get_session(5)
        summary = session.get_last_image_summary()
        assert "cat" in summary

    def test_multiple_users_isolated(self):
        self.sm.record_interaction(10, "text", "hello", "world")
        self.sm.record_interaction(11, "text", "foo", "bar")
        s10 = self.sm.get_session(10)
        s11 = self.sm.get_session(11)
        assert len(s10.history) == 1
        assert len(s11.history) == 1
        assert "hello" in s10.history[0].user_input
        assert "foo" in s11.history[0].user_input

    def test_stats(self):
        self.sm.get_session(20)
        self.sm.record_interaction(20, "text", "hi", "hello")
        stats = self.sm.get_all_stats()
        assert stats["total_users"] >= 1
        assert stats["users_with_history"] >= 1


# ──────────────────────────────────────────────────────────────────────────────
# Vision Service Tests (mocked model)
# ──────────────────────────────────────────────────────────────────────────────

class TestVisionService:
    def _make_service_with_mock(self, caption_output: str):
        """Create a VisionService with mocked model."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

        with patch("vision_service.VisionService._load_model"):
            from vision_service import VisionService
            svc = VisionService.__new__(VisionService)
            svc._image_hash_cache = {}
            svc._model_type = "vit-gpt2"
            svc._device = "cpu"
            svc._caption_vit_gpt2 = MagicMock(return_value=caption_output)
            svc._caption_blip2 = MagicMock(return_value=caption_output)
            return svc

    def test_describe_image_returns_caption(self):
        svc = self._make_service_with_mock("A beautiful sunset over mountains")
        result = svc.describe_image(make_test_image())
        assert result["caption"] == "A beautiful sunset over mountains"
        assert len(result["tags"]) == 3
        assert result["cached"] is False

    def test_caching_works(self):
        svc = self._make_service_with_mock("A dog playing in the park")
        image = make_test_image(color=(0, 255, 0))
        r1 = svc.describe_image(image)
        r2 = svc.describe_image(image)
        assert r1["cached"] is False
        assert r2["cached"] is True

    def test_different_images_not_cached(self):
        svc = self._make_service_with_mock("Some image")
        r1 = svc.describe_image(make_test_image(color=(255, 0, 0)))
        r2 = svc.describe_image(make_test_image(color=(0, 0, 255)))
        assert r1["cached"] is False
        assert r2["cached"] is False

    def test_question_skips_cache(self):
        svc = self._make_service_with_mock("A red car")
        image = make_test_image()
        r1 = svc.describe_image(image)
        r2 = svc.describe_image(image, question="What color is the car?")
        # Q&A should not return cached result
        assert r2["cached"] is False

    def test_extract_keywords_filters_stopwords(self):
        with patch("vision_service.VisionService._load_model"):
            from vision_service import VisionService
            svc = VisionService.__new__(VisionService)
            svc._image_hash_cache = {}
            keywords = svc._extract_keywords("A man and a woman are walking in the park")
            assert "a" not in keywords
            assert "the" not in keywords
            assert len(keywords) == 3

    def test_extract_keywords_pads_short_captions(self):
        with patch("vision_service.VisionService._load_model"):
            from vision_service import VisionService
            svc = VisionService.__new__(VisionService)
            svc._image_hash_cache = {}
            keywords = svc._extract_keywords("cat")
            assert len(keywords) == 3
            assert "image" in keywords  # padding

    def test_image_hash_is_consistent(self):
        with patch("vision_service.VisionService._load_model"):
            from vision_service import VisionService
            svc = VisionService.__new__(VisionService)
            svc._image_hash_cache = {}
            image_bytes = make_test_image()
            h1 = svc._compute_image_hash(image_bytes)
            h2 = svc._compute_image_hash(image_bytes)
            assert h1 == h2

    def test_cache_eviction(self):
        svc = self._make_service_with_mock("test")
        # Fill cache beyond limit
        for i in range(130):
            image = make_test_image(width=i + 10, height=i + 10)
            svc.describe_image(image)
        assert len(svc._image_hash_cache) <= 128

    def test_cache_stats(self):
        svc = self._make_service_with_mock("test")
        stats = svc.get_cache_stats()
        assert "cached_images" in stats
        assert "model_type" in stats
        assert "device" in stats


# ──────────────────────────────────────────────────────────────────────────────
# Config Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test123"}):
            from config import Config
            c = Config()
            assert c.MAX_HISTORY_PER_USER == 3
            assert c.CACHE_MAX_SIZE == 128
            assert c.GRADIO_PORT == 7860

    def test_validate_requires_token(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
        with patch.dict("os.environ", {}, clear=True):
            from config import Config
            c = Config()
            c.TELEGRAM_BOT_TOKEN = ""
            with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
                c.validate()


# ──────────────────────────────────────────────────────────────────────────────
# Handler Utility Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestHandlerUtils:
    def test_escape_md(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
        from handlers import escape_md
        result = escape_md("Hello! This is a test (with) special_chars.")
        assert "\\!" in result
        assert "\\(" in result
        assert "\\)" in result
        assert "\\_" in result
        assert "\\." in result

    def test_escape_md_plain_text(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
        from handlers import escape_md
        result = escape_md("Hello World")
        assert result == "Hello World"