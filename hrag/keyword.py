"""Keyword search: an in-memory BM25 index over the chunk texts.

Why not SQLite FTS5: Python's bundled SQLite does not guarantee FTS5 on every Windows build, and with a few hundred chunks
BM25 is instant and behaves identically everywhere.

The tokenizer keeps security identifiers searchable both whole and in parts: `CVE-2021-44228` is indexed as
`cve-2021-44228` AND `cve`, `2021`, `44228`; `/api/v1/users` as `api/v1/users` AND `api`, `v1`, `users`. Queries and documents
go through the same function, so they always agree.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

K1 = 1.5
B = 0.75
_WORD = re.compile(r"[a-z0-9_][a-z0-9_./:\-]*")
_PARTS = re.compile(r"[^a-z0-9]+")
_STOP = frozenset("a an and are as at be by for from in is it of on or that the this to was were with".split())


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _WORD.findall(text.lower()):
        whole = raw.rstrip(".:/-")
        if not whole:
            continue
        parts = [p for p in _PARTS.split(whole) if p]
        if whole not in _STOP:
            tokens.append(whole)
        if len(parts) > 1 or (parts and parts[0] != whole):
            tokens.extend(p for p in parts if p != whole and p not in _STOP)
    return tokens


class Bm25Index:
    def __init__(self, doc_ids: Sequence[int], texts: Sequence[str]):
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts have different lengths")
        self.doc_ids = list(doc_ids)
        self._tf = [Counter(tokenize(t)) for t in texts]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self._tf)
        self._idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def search(self, query: str, n: int) -> list[tuple[int, float]]:
        """(doc_id, BM25 score) for the n best documents that share at least one term with the query; ties by id."""
        terms = set(tokenize(query)) & self._idf.keys()
        if not terms:
            return []
        scored: list[tuple[int, float]] = []
        for doc_id, tf, length in zip(self.doc_ids, self._tf, self._len):
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if f:
                    score += self._idf[term] * f * (K1 + 1) / (f + K1 * (1 - B + B * length / (self._avg or 1.0)))
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored[:n]
