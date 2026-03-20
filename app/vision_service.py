"""
Vision Model Service
Handles image captioning using HuggingFace models.
Supports BLIP-2, LLaVA, and a lightweight fallback (ViT-GPT2).
"""

import hashlib
import logging
from io import BytesIO
from functools import lru_cache
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


class VisionService:
    """
    Wraps HuggingFace vision models for image captioning and tagging.

    Supported models:
    - Lightweight: nlpconnect/vit-gpt2-image-captioning  (~300MB, CPU-friendly)
    - Standard:    Salesforce/blip2-opt-2.7b             (~6GB, GPU recommended)
    - Advanced:    llava-hf/llava-1.5-7b-hf              (~14GB, GPU required)
    """

    def __init__(self, use_lightweight: bool = True, model_name: Optional[str] = None):
        self.use_lightweight = use_lightweight
        self.model_name = model_name
        self._processor = None
        self._model = None
        self._image_hash_cache: dict[str, dict] = {}  # hash -> result
        self._load_model()

    def _load_model(self):
        """Lazily load the vision model on first use."""
        try:
            if self.use_lightweight:
                self._load_lightweight_model()
            else:
                self._load_blip2_model()
        except Exception as e:
            logger.error(f"Failed to load primary model: {e}. Falling back to lightweight.")
            self._load_lightweight_model()

    def _load_lightweight_model(self):
        """Load ViT-GPT2 image captioning model (~300MB, CPU friendly)."""
        from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
        import torch

        model_id = "nlpconnect/vit-gpt2-image-captioning"
        logger.info(f"Loading lightweight model: {model_id}")

        self._processor = ViTImageProcessor.from_pretrained(model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model_type = "vit-gpt2"
        logger.info(f"Lightweight model loaded on {self._device}")

    def _load_blip2_model(self):
        """Load BLIP-2 model (~6GB, GPU recommended)."""
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import torch

        model_id = self.model_name or "Salesforce/blip2-opt-2.7b"
        logger.info(f"Loading BLIP-2 model: {model_id}")

        self._processor = Blip2Processor.from_pretrained(model_id)
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model_type = "blip2"
        logger.info(f"BLIP-2 model loaded on {self._device}")

    def _compute_image_hash(self, image_bytes: bytes) -> str:
        """Compute a SHA-256 hash of image bytes for caching."""
        return hashlib.sha256(image_bytes).hexdigest()

    def _extract_keywords(self, caption: str) -> list[str]:
        """
        Extract 3 meaningful keywords from the caption.
        Uses simple frequency + stop-word filtering.
        """
        import re

        STOP_WORDS = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "of", "in", "on", "at", "to", "for", "with", "by", "from",
            "and", "or", "but", "if", "as", "up", "it", "its", "this",
            "that", "there", "here", "what", "which", "who", "whom",
            "some", "two", "three", "very", "also", "not", "no",
        }

        words = re.findall(r"\b[a-zA-Z]{3,}\b", caption.lower())
        keywords = []
        seen = set()
        for w in words:
            if w not in STOP_WORDS and w not in seen:
                keywords.append(w)
                seen.add(w)
            if len(keywords) == 3:
                break

        # Pad if fewer than 3
        while len(keywords) < 3:
            keywords.append("image")

        return keywords

    def describe_image(self, image_bytes: bytes, question: Optional[str] = None) -> dict:
        """
        Generate a caption and tags for an image.

        Args:
            image_bytes: Raw image bytes.
            question: Optional question about the image (for VQA).

        Returns:
            dict with keys: caption, tags, cached (bool)
        """
        img_hash = self._compute_image_hash(image_bytes)

        # Cache hit
        if img_hash in self._image_hash_cache and question is None:
            logger.info("Cache hit for image hash.")
            result = self._image_hash_cache[img_hash].copy()
            result["cached"] = True
            return result

        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        try:
            if self._model_type == "vit-gpt2":
                caption = self._caption_vit_gpt2(image)
            elif self._model_type == "blip2":
                caption = self._caption_blip2(image, question)
            else:
                caption = "Unable to generate caption."
        except Exception as e:
            logger.error(f"Inference error: {e}")
            caption = "Sorry, I couldn't analyze this image."

        tags = self._extract_keywords(caption)
        result = {"caption": caption, "tags": tags, "cached": False}

        # Store in cache (only for pure captioning, not Q&A)
        if question is None:
            self._image_hash_cache[img_hash] = result.copy()
            # Evict oldest if over size limit
            if len(self._image_hash_cache) > 128:
                oldest_key = next(iter(self._image_hash_cache))
                del self._image_hash_cache[oldest_key]

        return result

    def _caption_vit_gpt2(self, image: Image.Image) -> str:
        """Run ViT-GPT2 captioning."""
        import torch

        pixel_values = self._processor(images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                pixel_values,
                max_length=64,
                num_beams=4,
                early_stopping=True,
            )

        caption = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip().capitalize()

    def _caption_blip2(self, image: Image.Image, question: Optional[str] = None) -> str:
        """Run BLIP-2 captioning or VQA."""
        import torch

        if question:
            prompt = f"Question: {question} Answer:"
            inputs = self._processor(image, text=prompt, return_tensors="pt").to(
                self._device, torch.float16 if "cuda" in str(self._device) else torch.float32
            )
        else:
            inputs = self._processor(image, return_tensors="pt").to(
                self._device, torch.float16 if "cuda" in str(self._device) else torch.float32
            )

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=100)

        caption = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption.strip().capitalize()

    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "cached_images": len(self._image_hash_cache),
            "model_type": getattr(self, "_model_type", "unknown"),
            "device": str(getattr(self, "_device", "unknown")),
        }