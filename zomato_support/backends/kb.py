"""
===============================================================================
KNOWLEDGE BASE - BM25 search over the policy markdown       [CORE, replaceable]
===============================================================================

WHAT THIS DOES
    Splits data/kb/*.md into one chunk per section and ranks them against a
    query with BM25 (a standard keyword-relevance formula).

WHY IT EXISTS
    The agent must not state policy from memory. It has to retrieve a chunk
    and cite it, which is what makes the guardrail's grounding check possible
    later - every number the agent quotes has to appear in something a tool
    actually returned.

WHY BM25 AND NOT A VECTOR STORE
    Zero dependencies and it fits in one file you can read. Swap it for a real
    vector store in production; the only interface that matters is

        search(query, k) -> [{chunk_id, title, text, score}, ...]

    Keep chunk_id on every result whatever you switch to - the citations are
    the point.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

_KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"

_TOKEN = re.compile(r"[a-z0-9]+")

# Cheap stoplist - enough to stop "the" dominating short policy chunks.
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "to was were will with not no do does can may must".split()
)

_K1 = 1.5
_B = 0.75


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class KnowledgeBase:
    def __init__(self, kb_dir: Path | None = None) -> None:
        self._chunks: list[dict[str, Any]] = []
        for path in sorted((kb_dir or _KB_DIR).glob("*.md")):
            self._chunks.extend(self._chunk_file(path))

        self._doc_tokens = [_tokenize(c["text"] + " " + c["title"]) for c in self._chunks]
        self._doc_freqs = [Counter(t) for t in self._doc_tokens]
        self._lengths = [len(t) for t in self._doc_tokens]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0

        n = len(self._chunks)
        df: Counter[str] = Counter()
        for tokens in self._doc_tokens:
            df.update(set(tokens))
        # BM25 idf with the +1 smoothing that keeps common terms non-negative.
        self._idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()
        }

    @staticmethod
    def _chunk_file(path: Path) -> list[dict[str, Any]]:
        """Split a policy file into one chunk per section.

        Each file's H1 carries the citable doc id (e.g. `KB-REFUND-001`), so a
        chunk can always be traced back to a line a human can look up.
        """
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()

        doc_id, doc_title = path.stem.upper(), path.stem
        if lines and lines[0].startswith("# "):
            heading = lines[0][2:].strip()
            doc_id, _, doc_title = heading.partition(":")
            doc_id, doc_title = doc_id.strip(), (doc_title.strip() or heading)

        body = "\n".join(lines[1:]).strip()
        sections = [s.strip() for s in re.split(r"\n\s*\n", body) if s.strip()]

        return [
            {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}#{i}",
                "title": doc_title,
                "text": section,
                "source": path.name,
            }
            for i, section in enumerate(sections)
        ]

    def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        terms = _tokenize(query)
        if not terms or not self._chunks:
            return []

        scored: list[tuple[float, int]] = []
        for i, freqs in enumerate(self._doc_freqs):
            length = self._lengths[i] or 1
            score = 0.0
            for term in terms:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                norm = tf * (_K1 + 1) / (
                    tf + _K1 * (1 - _B + _B * length / (self._avg_len or 1))
                )
                score += self._idf.get(term, 0.0) * norm
            if score > 0:
                scored.append((score, i))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [
            {**self._chunks[i], "score": round(score, 3)} for score, i in scored[:k]
        ]
