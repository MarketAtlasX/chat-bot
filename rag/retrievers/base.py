from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RetrieverType(str, Enum):
    NEWS = "news"
    MARKET = "market"
    HISTORICAL = "historical"
    GRAPH = "graph"


@dataclass
class RetrievalResult:
    content: str
    score: float
    source: str
    retriever_type: RetrieverType
    metadata: dict = field(default_factory=dict)
    id: str = ""


class BaseRetriever(ABC):
    def __init__(self, name: str, retriever_type: RetrieverType):
        self.name = name
        self.retriever_type = retriever_type

    @abstractmethod
    async def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[RetrievalResult]:
        ...
