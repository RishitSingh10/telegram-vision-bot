"""
Gradio UI for local debugging of the Vision Bot.
Run independently: python gradio_ui.py
"""

import gradio as gr
import logging
from io import BytesIO
from PIL import Image

from vision_service import VisionService
from session_manager import session_manager
from config import Config

logger = logging.getLogger(__name__)
config = Config()

# Load vision service
vision_service = VisionService(use_lightweight=config.USE_LIGHTWEIGHT_MODEL)


def analyze_image(image: Image.Image, question: str, user_id: str) -> tuple[str, str, str, str]:
    """
    Analyze image and return caption, tags, cache status, and history.

    Returns: (caption, tags, cache_status, history)
    """
    if image is None:
        return "No image provided.", "", "", "No history."

    # Convert PIL Image to bytes
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()

    uid = int(user_id) if user_id.isdigit() else 9999
    question_text = question.strip() if question.strip() else None

    result = vision_service.describe_image(image_bytes, question=question_text)

    caption = result["caption"]
    tags = result["tags"]
    cached = result["cached"]

    tags_str = "  ".join(f"#{t}" for t in tags)
    cache_status = "⚡ Cached result" if cached else "🔄 Freshly analyzed"

    # Update session
    session_manager.store_last_image(uid, image_bytes)
    session_manager.record_interaction(
        user_id=uid,
        interaction_type="image",
        user_input=question_text or "[image]",
        bot_response=f"Caption: {caption}",
        caption=caption,
        tags=tags,
    )

    # Get history
    session = session_manager.get_session(uid)
    history = session.get_history_text()

    return caption, tags_str, cache_status, history


def get_summary(user_id: str) -> str:
    """Return session summary for a user."""
    uid = int(user_id) if user_id.isdigit() else 9999
    session = session_manager.get_session(uid)
    img_summary = session.get_last_image_summary()
    history = session.get_history_text()
    return f"{img_summary}\n\n--- Recent History ---\n{history}"


def get_cache_stats() -> str:
    stats = vision_service.get_cache_stats()
    sm_stats = session_manager.get_all_stats()
    return (
        f"Model: {stats['model_type']}\n"
        f"Device: {stats['device']}\n"
        f"Cached images: {stats['cached_images']}\n"
        f"Total users: {sm_stats['total_users']}\n"
        f"Users with history: {sm_stats['users_with_history']}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Build Gradio interface
# ──────────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Vision Bot Debugger",
    theme=gr.themes.Soft(primary_hue="indigo"),
    css="""
        .header { text-align: center; margin-bottom: 1em; }
        .result-box { font-size: 1.1em; }
    """,
) as demo:

    gr.Markdown(
        """
        # 🤖 Vision Bot — Local Debugger
        Test the image captioning pipeline locally before deploying to Telegram.
        """,
        elem_classes=["header"],
    )

    with gr.Tabs():
        # ── Tab 1: Image Analysis ──────────────────────────────────────────
        with gr.Tab("📸 Image Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Upload Image")
                    question_input = gr.Textbox(
                        label="Ask a question (optional)",
                        placeholder="e.g. What color is the car?",
                    )
                    user_id_input = gr.Textbox(
                        label="User ID (for session tracking)",
                        value="1001",
                        max_lines=1,
                    )
                    analyze_btn = gr.Button("🔍 Analyze Image", variant="primary")

                with gr.Column(scale=1):
                    caption_out = gr.Textbox(
                        label="📝 Caption",
                        interactive=False,
                        elem_classes=["result-box"],
                    )
                    tags_out = gr.Textbox(
                        label="🏷 Tags",
                        interactive=False,
                    )
                    cache_out = gr.Textbox(
                        label="⚡ Cache Status",
                        interactive=False,
                    )
                    history_out = gr.Textbox(
                        label="📋 Session History",
                        interactive=False,
                        lines=6,
                    )

            analyze_btn.click(
                fn=analyze_image,
                inputs=[image_input, question_input, user_id_input],
                outputs=[caption_out, tags_out, cache_out, history_out],
            )

        # ── Tab 2: Session Summary ─────────────────────────────────────────
        with gr.Tab("📋 Session Summary"):
            with gr.Row():
                uid_input2 = gr.Textbox(label="User ID", value="1001")
                summarize_btn = gr.Button("📋 Get Summary", variant="secondary")
            summary_out = gr.Textbox(label="Summary", lines=10, interactive=False)

            summarize_btn.click(
                fn=get_summary,
                inputs=[uid_input2],
                outputs=[summary_out],
            )

        # ── Tab 3: Stats ───────────────────────────────────────────────────
        with gr.Tab("📊 Stats"):
            stats_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
            stats_out = gr.Textbox(label="System Stats", lines=8, interactive=False)

            stats_btn.click(fn=get_cache_stats, inputs=[], outputs=[stats_out])

            # Load stats on startup
            demo.load(fn=get_cache_stats, outputs=[stats_out])

    gr.Markdown(
        """
        ---
        💡 **Tip:** Upload the same image twice to see caching in action!
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name=config.GRADIO_HOST,
        server_port=config.GRADIO_PORT,
        share=False,
    )