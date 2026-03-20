# 🤖 Telegram Vision Bot

A lightweight GenAI Telegram bot that accepts image uploads, generates captions and keyword tags using open-source HuggingFace vision models, and maintains per-user conversation history.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📸 Image captioning | Short natural-language description per image |
| 🏷 Keyword tagging | 3 auto-extracted tags per image |
| 💬 Message history | Last 3 interactions stored per user |
| ⚡ Image caching | SHA-256 hash-based — repeated images are free |
| 📋 /summarize | Recap of last image + conversation history |
| 🖥 Gradio UI | Local debug dashboard at `localhost:7860` |
| 🐳 Docker support | Single command to spin up everything |

---

## 🏗 System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        TELEGRAM CLIENT                          │
│              (User sends image / command / text)                │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTPS (Telegram Bot API)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       BOT APPLICATION                           │
│                       (bot.py / app.py)                         │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │   /start     │   │   /help      │   │   /summarize       │  │
│  │   /image     │   │  Text msgs   │   │   Photo handler    │  │
│  └──────────────┘   └──────────────┘   └─────────┬──────────┘  │
│                                                   │             │
│              handlers.py                          │             │
└───────────────────────────────────────────────────┼─────────────┘
                                                    │
                    ┌───────────────────────────────┤
                    │                               │
                    ▼                               ▼
     ┌──────────────────────────┐    ┌──────────────────────────┐
     │      VisionService       │    │     SessionManager       │
     │     (vision_service.py)  │    │   (session_manager.py)   │
     │                          │    │                          │
     │  • Image → bytes         │    │  • Per-user history      │
     │  • SHA-256 hash          │    │  • deque(maxlen=3)        │
     │  • Cache lookup ────────►│    │  • last_image store      │
     │  • Model inference       │    │  • awaiting_image flag   │
     │  • Caption + tags        │    │                          │
     └──────────┬───────────────┘    └──────────────────────────┘
                │
                ▼
     ┌──────────────────────────┐
     │    HuggingFace Model     │
     │                          │
     │  LIGHTWEIGHT (default):  │
     │  nlpconnect/             │
     │  vit-gpt2-image-         │
     │  captioning (~300MB)     │
     │                          │
     │  STANDARD:               │
     │  Salesforce/blip2-       │
     │  opt-2.7b (~6GB)         │
     │                          │
     │  ADVANCED:               │
     │  llava-hf/               │
     │  llava-1.5-7b-hf (~14GB) │
     └──────────────────────────┘

     ┌──────────────────────────┐
     │      Gradio Debug UI     │
     │     (gradio_ui.py)       │
     │   localhost:7860         │
     │                          │
     │  Tab 1: Image Analysis   │
     │  Tab 2: Session Summary  │
     │  Tab 3: Cache Stats      │
     └──────────────────────────┘
```

### Data Flow for Image Analysis

```
User sends photo
      │
      ▼
photo_handler() called
      │
      ├─► Download largest photo from Telegram
      │
      ├─► Compute SHA-256 hash of image bytes
      │
      ├─► Cache hit? ──YES──► Return cached {caption, tags}
      │        │
      │       NO
      │        ▼
      ├─► VisionService.describe_image()
      │        │
      │        ▼
      │   Model inference
      │   (ViT-GPT2 or BLIP-2)
      │        │
      │        ▼
      │   Extract 3 keywords
      │   (stopword filter)
      │        │
      │        ▼
      │   Store in cache
      │
      ├─► SessionManager.record_interaction()
      │   (deque maxlen=3 per user)
      │
      └─► Reply to user with caption + hashtags
```

---

## 🚀 How to Run Locally

### Prerequisites

- Python 3.11+
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- ~500MB disk space (lightweight model) or ~6–14GB (BLIP-2/LLaVA)

### 1. Clone and set up

```bash
git clone <repo>
cd telegram-vision-bot

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set your TELEGRAM_BOT_TOKEN
```

### 3. Run

```bash
# Bot only (Telegram)
python app.py

# Bot + Gradio debug UI
python app.py --ui

# Gradio UI only (no Telegram needed)
python app.py --ui-only
```

The Gradio UI is available at: **http://localhost:7860**

### 4. Run tests

```bash
pip install pytest
pytest tests/ -v
```

---

## 🐳 Docker Compose

```bash
# Copy and edit your .env
cp .env.example .env

# Start everything
docker compose up --build

# Bot only
docker compose up bot

# Gradio UI only
docker compose up gradio
```

The `hf_cache` Docker volume persists downloaded model weights across container restarts.

---

## 🤖 Bot Commands

| Command | Description |
|---|---|
| `/start` | Welcome message |
| `/help` | Usage instructions |
| `/image` | Prompt the bot to expect an image |
| `/summarize` | Recap last image + recent chat |

You can also **send an image directly** without any command.
Add a caption to the image to ask a specific question (VQA mode with BLIP-2).

---

## 🧠 Models Used

### Lightweight (default, CPU-friendly)
- **`nlpconnect/vit-gpt2-image-captioning`**
  - Size: ~300MB
  - Architecture: ViT encoder + GPT-2 decoder
  - CPU inference: ~2–5 seconds per image
  - Ideal for: demos, low-resource machines

### Standard (GPU recommended)
- **`Salesforce/blip2-opt-2.7b`**
  - Size: ~6GB
  - Architecture: BLIP-2 with OPT-2.7B language model
  - Supports VQA (question about image in caption)
  - GPU inference: ~1–3 seconds per image

### Advanced (GPU required)
- **`llava-hf/llava-1.5-7b-hf`**
  - Size: ~14GB
  - Architecture: LLaVA multimodal with Vicuna-7B
  - Best conversational image understanding
  - GPU inference: ~3–8 seconds per image

To switch models, set in `.env`:
```bash
USE_LIGHTWEIGHT_MODEL=false
VISION_MODEL=Salesforce/blip2-opt-2.7b
```

---

## ⚙️ Architecture Decisions

### Caching Strategy
- Images are hashed with SHA-256 before inference
- Cache is an in-memory Python dict (LRU-style eviction at 128 entries)
- Trade-off: fast lookups, lost on restart — suitable for single-instance deployment
- Production upgrade: Redis for persistent, multi-instance cache

### History Management
- `collections.deque(maxlen=3)` per user — O(1) append, auto-eviction
- Stored in-memory in `SessionManager` singleton
- Production upgrade: Redis or SQLite for persistence across restarts

### Model Loading
- Models are loaded once on first use (lazy init)
- HuggingFace caches weights in `~/.cache/huggingface` (or `HF_HOME`)
- Docker mounts a named volume to persist weights

---

## 📁 Project Structure

```
telegram-vision-bot/
├── app/
│   ├── bot.py              # Telegram Application setup + polling
│   ├── handlers.py         # Command & message handlers
│   ├── vision_service.py   # HuggingFace model wrapper + cache
│   ├── session_manager.py  # Per-user history & state
│   ├── gradio_ui.py        # Local debug UI
│   └── config.py           # Environment-based config
├── tests/
│   ├── conftest.py
│   └── test_bot.py         # Unit tests (18 tests)
├── app.py                  # Unified entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 🔧 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | *(required)* | From [@BotFather](https://t.me/BotFather) |
| `USE_LIGHTWEIGHT_MODEL` | `true` | Use ViT-GPT2 (CPU) vs BLIP-2 (GPU) |
| `VISION_MODEL` | `Salesforce/blip2-opt-2.7b` | HF model ID when not lightweight |
| `GRADIO_PORT` | `7860` | Port for Gradio debug UI |
| `GRADIO_HOST` | `0.0.0.0` | Host for Gradio |
| `LOG_LEVEL` | `INFO` | Python logging level |