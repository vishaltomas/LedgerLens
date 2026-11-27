import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)


class Embed:
    """Load an embedding model once, then count tokens and embed chunks and queries."""

    def __init__(self, model_name: str = settings.embedding_model_name, device: str | None = None):
        cuda_ok = torch.cuda.is_available()
        if device not in ("cuda", "cpu"):
            logger.warning("Device %r not available, choosing automatically", device or "")
            device = "cuda" if cuda_ok else "cpu"
        self.device = device

        self.model = SentenceTransformer(model_name, device=self.device)
        self.tokenizer = self.model.tokenizer
        self.max_tokens = self.model.max_seq_length
        # [CLS] and [SEP] are added to every text, so a chunk gets slightly less room
        self.max_chunk_tokens = self.max_tokens - self.tokenizer.num_special_tokens_to_add()
        self.dim = self.model.get_embedding_dimension()

        # Logged so you can check it matches the model card (bge: cls)
        pooling = self._pooling_mode()
        logger.info(
            "Loaded %s on %s (pooling=%s, max_tokens=%d, dim=%d)",
            model_name,
            self.device,
            pooling,
            self.max_tokens,
            self.dim,
        )

    def _pooling_mode(self) -> str | None:
        for module in self.model:
            config = module.get_config_dict() if hasattr(module, "get_config_dict") else {}
            if "pooling_mode" in config:
                return config["pooling_mode"]
        return None

    def count_tokens(self, text: str) -> int:
        """Tokens in `text`, excluding [CLS]/[SEP]; compare with `max_chunk_tokens`."""
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed chunks. Returns float32 array of shape (len(texts), dim), rows normalized."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        too_long_texts = sum(self.count_tokens(t) > self.max_chunk_tokens for t in texts)
        if too_long_texts:
            logger.warning(
                "%d of %d texts exceed %d tokens and will be truncated",
                too_long_texts,
                len(texts),
                self.max_chunk_tokens,
            )
        return self.model.encode_document(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > batch_size,
        )

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one search query with the model's query prompt. Returns shape (dim,)."""
        return self.model.encode_query(text, normalize_embeddings=True)
