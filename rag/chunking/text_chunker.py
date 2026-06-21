from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    TOKEN = "token"
    SEMANTIC = "semantic"


@dataclass
class ChunkResult:
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    def chunk(self, text: str, strategy: Optional[ChunkStrategy] = None, metadata: Optional[dict] = None) -> List[ChunkResult]:
        strategy = strategy or self.strategy
        metadata = metadata or {}
        dispatch = {
            ChunkStrategy.RECURSIVE: self._chunk_recursive,
            ChunkStrategy.SENTENCE: self._chunk_sentence,
            ChunkStrategy.PARAGRAPH: self._chunk_paragraph,
            ChunkStrategy.TOKEN: self._chunk_token,
            ChunkStrategy.SEMANTIC: self._chunk_semantic,
        }
        return dispatch[strategy](text, metadata)

    def _chunk_recursive(self, text: str, metadata: dict) -> List[ChunkResult]:
        separators = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                best_split = -1
                for sep in separators:
                    pos = text.rfind(sep, start, end)
                    if pos > best_split:
                        best_split = pos
                if best_split > start:
                    end = best_split + len(separators[0]) if separators[0] and text[best_split:best_split+len(separators[0])] == separators[0] else best_split
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(ChunkResult(
                    text=chunk_text,
                    index=idx,
                    start_char=start,
                    end_char=end,
                    metadata={**metadata, "strategy": "recursive"},
                ))
                idx += 1
            start = end - self.chunk_overlap if end < len(text) else end
        return chunks

    def _chunk_sentence(self, text: str, metadata: dict) -> List[ChunkResult]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = []
        current_len = 0
        idx = 0
        pos = 0
        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > self.chunk_size and current:
                chunk_text = " ".join(current)
                chunks.append(ChunkResult(
                    text=chunk_text,
                    index=idx,
                    start_char=pos - len(chunk_text),
                    end_char=pos,
                    metadata={**metadata, "strategy": "sentence"},
                ))
                idx += 1
                overlap_count = max(1, len(current) // 2)
                current = current[-overlap_count:]
                current_len = sum(len(s) for s in current)
            current.append(sent)
            current_len += sent_len
            pos += len(sent) + 1
        if current:
            chunk_text = " ".join(current)
            chunks.append(ChunkResult(
                text=chunk_text,
                index=idx,
                start_char=pos - len(chunk_text),
                end_char=pos,
                metadata={**metadata, "strategy": "sentence"},
            ))
        return chunks

    def _chunk_paragraph(self, text: str, metadata: dict) -> List[ChunkResult]:
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current = []
        current_len = 0
        idx = 0
        pos = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if current_len + len(para) > self.chunk_size and current:
                chunk_text = "\n\n".join(current)
                chunks.append(ChunkResult(
                    text=chunk_text,
                    index=idx,
                    start_char=pos - len(chunk_text),
                    end_char=pos,
                    metadata={**metadata, "strategy": "paragraph"},
                ))
                idx += 1
                overlap_count = max(1, len(current) // 2)
                current = current[-overlap_count:]
                current_len = sum(len(p) for p in current)
            current.append(para)
            current_len += len(para)
            pos += len(para) + 2
        if current:
            chunk_text = "\n\n".join(current)
            chunks.append(ChunkResult(
                text=chunk_text,
                index=idx,
                start_char=pos - len(chunk_text),
                end_char=pos,
                metadata={**metadata, "strategy": "paragraph"},
            ))
        return chunks

    def _chunk_token(self, text: str, metadata: dict) -> List[ChunkResult]:
        words = text.split()
        chunks = []
        idx = 0
        pos = 0
        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            start = text.find(chunk_text, pos)
            if start == -1:
                start = pos
            end = start + len(chunk_text)
            chunks.append(ChunkResult(
                text=chunk_text,
                index=idx,
                start_char=start,
                end_char=end,
                metadata={**metadata, "strategy": "token"},
            ))
            idx += 1
            pos = end
            overlap = min(self.chunk_overlap, len(words) - i - self.chunk_size)
            i += self.chunk_size - overlap if overlap > 0 else self.chunk_size
        return chunks

    def _chunk_semantic(self, text: str, metadata: dict) -> List[ChunkResult]:
        return self._chunk_recursive(text, {**metadata, "strategy": "semantic"})
