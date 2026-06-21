from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache"


class BGEMModel:
    def __init__(self, model_name: str = "BAAI/bge-m3", cache_dir: Optional[Path] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir or CACHE_DIR
        self._model = None
        self._available = False
        self._load_model()

    def _load_model(self):
        cache_str = str(self.cache_dir)
        os.makedirs(cache_str, exist_ok=True)
        os.environ["HF_HUB_CACHE"] = cache_str
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_str
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, cache_folder=cache_str)
            self._available = True
            logger.info(f"BGE-M3 model loaded from {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load BGE-M3: {e}. Using fallback.")

    @property
    def available(self) -> bool:
        return self._available

    @property
    def dim(self) -> int:
        return 1024

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if self._available:
            embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.array(embeddings, dtype=np.float32)
        rng = np.random.RandomState(42)
        return np.array([rng.randn(self.dim) for _ in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed(text).flatten()


_model_instance: Optional[BGEMModel] = None


def get_embedding_model(force_reload: bool = False) -> BGEMModel:
    global _model_instance
    if _model_instance is None or force_reload:
        _model_instance = BGEMModel()
    return _model_instance


def get_available_models() -> list:
    return ["BAAI/bge-m3", "BAAI/bge-small-en-v1.5", "all-MiniLM-L6-v2"]
