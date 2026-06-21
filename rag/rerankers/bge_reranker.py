from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    content: str
    score: float
    original_index: int
    metadata: dict = field(default_factory=dict)
    id: str = ""


class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._available = False
        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._available = True
            logger.info(f"BGE Reranker loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load BGE reranker {self.model_name}: {e}. Using fallback.")

    @property
    def available(self) -> bool:
        return self._available

    def rerank(
        self,
        query: str,
        documents: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        if not documents:
            return []
        if self._available:
            try:
                import torch
                pairs = [[query, doc] for doc in documents]
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                )
                with torch.no_grad():
                    scores = self._model(**inputs).logits.view(-1).float().numpy()
                indexed = [
                    (i, float(score))
                    for i, score in enumerate(scores)
                ]
                indexed.sort(key=lambda x: x[1], reverse=True)
                if top_k:
                    indexed = indexed[:top_k]
                results = []
                for orig_idx, score in indexed:
                    results.append(
                        RerankResult(
                            content=documents[orig_idx],
                            score=score,
                            original_index=orig_idx,
                            metadata=metadatas[orig_idx] if metadatas and orig_idx < len(metadatas) else {},
                            id=ids[orig_idx] if ids and orig_idx < len(ids) else "",
                        )
                    )
                return results
            except Exception as e:
                logger.warning(f"Reranker inference failed: {e}. Using fallback.")

        indexed = list(enumerate(documents))
        indexed.sort(key=lambda x: len(set(query.lower().split()) & set(x[1].lower().split())), reverse=True)
        if top_k:
            indexed = indexed[:top_k]
        results = []
        for orig_idx, doc in indexed:
            overlap = len(set(query.lower().split()) & set(doc.lower().split()))
            score = overlap / max(len(query.split()), 1)
            results.append(
                RerankResult(
                    content=doc,
                    score=score,
                    original_index=orig_idx,
                    metadata=metadatas[orig_idx] if metadatas and orig_idx < len(metadatas) else {},
                    id=ids[orig_idx] if ids and orig_idx < len(ids) else "",
                )
            )
        return results


_reranker_instance: Optional[BGEReranker] = None


def get_reranker() -> BGEReranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = BGEReranker()
    return _reranker_instance
