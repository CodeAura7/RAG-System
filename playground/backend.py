"""Backend of the RAG Playground. Everything that matters is REUSED from the `hrag` package:

  retrieval        hrag.retrieve.Retriever.search           (vector + BM25 + Reciprocal Rank Fusion, dedupe, per-report cap)
  query embedding  hrag.cli.make_retriever -> hrag.embeddings.Embedder  (the profile in config.toml, e.g. OpenRouter 3-large)
  answering        hrag.answer.answer_question               (same prompt, same JSON contract, same citation parsing)
  judging          hrag.evaluate.judge_answer                (optional)
  metrics          hrag.evaluate.recall_at_k / reciprocal_rank / ndcg_at_k / report_ranking

What this file adds is only presentation data and bookkeeping: per-leg scores and ranks for the "why was this retrieved" view,
a cache so one question costs one embedding call, timings, display-only citation renumbering ([C123] -> [1]), the chunk
heuristics, and an append-only JSONL file for your relevance marks. It never writes to the database.

Two underscore-prefixed Retriever methods (`_vector_ranked`, `_bm25_ranked`) are called to read the per-leg scores that
`Retriever.search` computes internally but does not return. They are the exact calls `search` makes, with the same arguments;
tests/test_playground_backend.py asserts that their ranks equal the `via` ranks `search` reports, so a change to the retriever
that breaks this fails loudly instead of showing wrong numbers.
"""
from __future__ import annotations

import csv
import difflib
import io
import json
import re
import statistics
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from hrag import cli
from hrag.answer import AnswerResult, answer_question
from hrag.config import Config, load_config
from hrag.evaluate import Question, judge_answer, ndcg_at_k, recall_at_k, reciprocal_rank, report_ranking
from hrag.keyword import Bm25Index
from hrag.retrieve import Hit, Retriever
from hrag.secrets import get_secret
from hrag.store import ChunkRow, ReportRow, Store

MODES = ("hybrid", "vector", "bm25")
LABELS = ("unjudged", "relevant", "irrelevant")
ACCURACY = ("not rated", "correct", "partially correct", "incorrect", "abstained - correctly", "abstained - wrongly")
FAITHFULNESS = ("not rated", "fully supported by the sources", "some claims unsupported", "mostly unsupported")
KS = (1, 3, 5, 10)
SCHEMA_VERSION = 1


class PlaygroundError(RuntimeError):
    pass


# ---------- small pure helpers ----------

_GROUP = re.compile(r"\[\s*((?:[Cc]\d+\s*[,;]?\s*)+)\]")
_ID = re.compile(r"[Cc](\d+)")


def inline_citation_ids(text: str) -> list[int]:
    """Chunk ids written inline as [C12] or [C12, C15], in order of first appearance."""
    ids = [int(i) for g in _GROUP.findall(text) for i in _ID.findall(g)]
    return list(dict.fromkeys(ids))


def renumber_citations(text: str, supplied_chunk_ids: Sequence[int]) -> str:
    """DISPLAY ONLY: [C12] -> [2] when chunk 12 was the 2nd source shown to the model. Unknown ids become [C99?] so a
    citation of something the model was never shown cannot pass for a real one. The model call itself is untouched."""
    number = {cid: i for i, cid in enumerate(supplied_chunk_ids, start=1)}

    def swap(m: re.Match) -> str:
        parts = [str(number[int(i)]) if int(i) in number else f"C{i}?" for i in _ID.findall(m.group(1))]
        return "[" + ", ".join(parts) + "]"

    return _GROUP.sub(swap, text)


def answer_notice(result: AnswerResult) -> tuple[str, str] | None:
    """What the page should say about how the model's reply was read: (level, text), or None when it was clean JSON."""
    if result.parse_error:
        return ("warning", "The model did not return the requested JSON, so its citations could not be checked. "
                           "Below is its reply exactly as received.")
    if result.parse_how == "repaired":
        return ("info", "The model's reply was not clean JSON (for example citation ids without quotes), so it was repaired. "
                        "The citations were read and are checked against the sources shown.")
    if result.parse_how == "salvaged":
        return ("info", "The model's reply was badly formed JSON. The answer text and citations were recovered from it as well as "
                        "possible; treat the citations with care.")
    return None


def safe_markdown(text: str) -> str:
    """The answer is model text built from untrusted reports. Streamlit renders markdown images by fetching their URL,
    which would let a report make your browser request an attacker's address; neutralise the image syntax."""
    return text.replace("![", "!\\[")


def precision_at_k(ranking: Sequence[int], gold: set[int], k: int) -> float:
    return len(set(ranking[:k]) & gold) / k if k > 0 else 0.0


def _mean(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def key_available(name: str) -> bool:
    return bool(get_secret(name))


# ---------- query embedding: one paid call per distinct question, with timing ----------

class CachingQueryEmbedder:
    """Wraps the real query embedder. The playground asks the retriever for the answer AND for the per-leg scores, and a
    comparison run asks the same question under several chunk sets; none of that should pay for the embedding twice."""

    def __init__(self, inner: Any, cache: dict, profile: str):
        self.inner, self.cache, self.profile = inner, cache, profile
        self.last_ms = 0.0
        self.last_cached = False

    def embed_query(self, text: str):
        key = (self.profile, text)
        if key in self.cache:
            self.last_ms, self.last_cached = 0.0, True
            return self.cache[key][0]
        t0 = time.perf_counter()
        vec = self.inner.embed_query(text)
        ms = (time.perf_counter() - t0) * 1000
        self.cache[key] = (vec, ms)
        self.last_ms, self.last_cached = ms, False
        return vec


# ---------- result objects ----------

@dataclass
class ChunkView:
    rank: int
    chunk_id: int
    report_id: int
    title: str
    program: str
    weakness: str
    section: str
    ordinal: int
    n_tokens: int
    has_code: bool
    header: str
    text: str
    score: float
    source_url: str
    supplied: bool                      # True = among the top K that the answer model is shown
    vector_rank: int | None = None
    vector_score: float | None = None   # cosine similarity
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_vector: float | None = None     # this leg's contribution to the fused score: 1 / (rrf_k + rank)
    rrf_bm25: float | None = None


@dataclass
class QueryRun:
    id: str
    question: str
    chunk_set: str
    mode: str
    k: int
    pool: int
    rerank: bool
    embed_profile: str
    rrf_k: int
    views: list[ChunkView]
    hits: list[Hit]                     # the real Hit objects; the first `k` are what the answer model receives
    timings: dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    scope_report_id: int | None = None  # set when retrieval was restricted to one report (the Report Library)

    @property
    def supplied(self) -> list[ChunkView]:
        return [v for v in self.views if v.supplied]

    @property
    def extra(self) -> list[ChunkView]:
        return [v for v in self.views if not v.supplied]


@dataclass
class AnswerRun:
    result: AnswerResult
    display_text: str
    cited_valid: list[int]              # cited AND among the chunks that were shown to the model
    cited_invalid: list[int]            # cited but never shown (the model made it up)
    inline_ids: list[int]               # ids written in the answer text itself
    llm_ms: float
    model: str


# ---------- the backend ----------

class Backend:
    def __init__(self, cfg: Config, *, retriever_factory: Callable[..., Retriever] | None = None,
                 llm_factory: Callable[[str], Any] | None = None):
        self.cfg = cfg
        self._retriever_factory = retriever_factory or self._default_retriever
        self._llm_factory = llm_factory or (lambda name: cli.make_llm(cfg, name))
        self._retrievers: dict[tuple, Retriever] = {}
        self._embed_cache: dict[tuple[str, str], tuple[Any, float]] = {}
        self._reports: tuple[int, dict[int, ReportRow]] | None = None
        self._originals: tuple[int, dict[int, str]] | None = None

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "Backend":
        return cls(load_config(path or None))

    # ---- database (read only use) ----
    def _mtime(self) -> int:
        try:
            return self.cfg.db_path.stat().st_mtime_ns
        except OSError as exc:
            raise PlaygroundError(f"database not found at {self.cfg.db_path}. Build it first (ingest-dataset / collect, then build); "
                                  "the playground never creates or replaces a database.") from exc

    def open_store(self) -> Store:
        self._mtime()                                   # refuse to create an empty database by accident
        return Store(self.cfg.db_path)

    def overview(self) -> dict[str, Any]:
        store = self.open_store()
        try:
            return store.stats()
        finally:
            store.close()

    def reports(self) -> dict[int, ReportRow]:
        stamp = self._mtime()
        if self._reports is None or self._reports[0] != stamp:
            store = self.open_store()
            try:
                self._reports = (stamp, {r.report_id: r for r in store.reports()})
            finally:
                store.close()
        return self._reports[1]

    def originals(self) -> dict[int, str]:
        """The dataset's own text of every report in the database, read from the cached parquet files ({} when they are absent)."""
        stamp = self._mtime()
        if self._originals is None or self._originals[0] != stamp:
            self._originals = (stamp, read_original_texts(self.cfg, set(self.reports())))
        return self._originals[1]

    def chunks(self, chunk_set: str) -> list[ChunkRow]:
        store = self.open_store()
        try:
            return store.chunks(chunk_set)
        finally:
            store.close()

    def chunk_set_config(self, chunk_set: str) -> dict | None:
        store = self.open_store()
        try:
            return store.chunk_set_config(chunk_set)
        finally:
            store.close()

    def embedded_count(self, chunk_set: str, profile: str) -> int:
        store = self.open_store()
        try:
            return len(store.embedded_ids(chunk_set, profile))
        finally:
            store.close()

    # ---- retriever (cached; it holds no database handle after construction) ----
    def _default_retriever(self, store: Store, chunk_set: str, embed_profile: str | None, need_vectors: bool, need_rerank: bool) -> Retriever:
        return cli.make_retriever(self.cfg, store, chunk_set, embed_profile, need_vectors=need_vectors, need_rerank=need_rerank)

    def retriever(self, chunk_set: str, embed_profile: str | None, *, need_vectors: bool, need_rerank: bool) -> Retriever:
        stamp = self._mtime()
        profile = embed_profile or self.cfg.defaults.embedding
        key = (chunk_set, profile if need_vectors else None, need_vectors, need_rerank, stamp)
        if key not in self._retrievers:
            self._retrievers = {k: v for k, v in self._retrievers.items() if k[-1] == stamp}      # drop ones built from an older database
            store = self.open_store()
            try:
                r = self._retriever_factory(store, chunk_set, profile if need_vectors else None, need_vectors, need_rerank)
            finally:
                store.close()
            if r.embedder is not None:
                r.embedder = CachingQueryEmbedder(r.embedder, self._embed_cache, profile)
            self._retrievers[key] = r
        return self._retrievers[key]

    # ---- retrieval ----
    def retrieve(self, question: str, *, chunk_set: str, mode: str, k: int = 5, pool: int = 20, rerank: bool = False,
                 embed_profile: str | None = None, report_id: int | None = None) -> QueryRun:
        question = question.strip()
        if not question:
            raise PlaygroundError("type a question first")
        if mode not in MODES:
            raise PlaygroundError(f"mode must be one of {MODES}")
        need_vectors = mode in ("vector", "hybrid")
        profile = embed_profile or self.cfg.defaults.embedding
        pool = max(pool, k)
        # Inside ONE report the "at most N chunks per report" cap would otherwise cut every result list to N, so it is lifted to the depth.
        settings = cli.settings_with(self.cfg, mode=mode, top_k=k, rerank=True if rerank else None,
                                     max_chunks_per_report=pool if report_id is not None else None)
        r = self.retriever(chunk_set, profile, need_vectors=need_vectors, need_rerank=rerank)
        if report_id is not None:
            r = ScopedRetriever(r, report_id)
        emb = r.embedder if isinstance(r.embedder, CachingQueryEmbedder) else None

        t0 = time.perf_counter()
        hits = r.search(question, settings, mode=mode, top_k=pool, rerank=rerank)           # THE project's retrieval
        total_ms = (time.perf_counter() - t0) * 1000
        embed_ms = emb.last_ms if emb and need_vectors else 0.0
        cached = bool(emb.last_cached) if emb and need_vectors else False
        first_ms = self._embed_cache.get((profile, question), (None, None))[1] if need_vectors else None

        # per-leg scores: the same two calls Retriever.search makes, same n, embedding served from the cache above
        n = max(settings.candidates, pool)
        vec = {cid: (i, s) for i, (cid, s) in enumerate(r._vector_ranked(question, n), start=1)} if need_vectors else {}
        bm = {cid: (i, s) for i, (cid, s) in enumerate(r._bm25_ranked(question, n), start=1)} if mode in ("bm25", "hybrid") else {}

        reports = self.reports()
        views = []
        for h in hits:
            c = h.chunk
            rep = reports.get(c.report_id)
            v_rank, v_score = vec.get(c.chunk_id, (None, None))
            b_rank, b_score = bm.get(c.chunk_id, (None, None))
            views.append(ChunkView(
                rank=h.rank, chunk_id=c.chunk_id, report_id=c.report_id,
                title=(rep.title if rep else c.meta.get("title", "")), program=(rep.program if rep else c.meta.get("program", "")),
                weakness=(rep.weakness if rep else c.meta.get("weakness", "")), section=c.section, ordinal=c.ordinal,
                n_tokens=c.n_tokens, has_code=c.has_code, header=c.header, text=c.text, score=h.score,
                source_url=(rep.source_url if rep and rep.source_url else f"https://hackerone.com/reports/{c.report_id}"),
                supplied=h.rank <= k, vector_rank=v_rank, vector_score=v_score, bm25_rank=b_rank, bm25_score=b_score,
                rrf_vector=(1.0 / (settings.rrf_k + v_rank) if mode == "hybrid" and v_rank else None),
                rrf_bm25=(1.0 / (settings.rrf_k + b_rank) if mode == "hybrid" and b_rank else None)))
        timings = {"total_ms": round(total_ms, 1), "embed_ms": round(embed_ms, 1), "search_ms": round(max(total_ms - embed_ms, 0.0), 1),
                   "embed_cached": cached, "embed_first_ms": round(first_ms, 1) if first_ms else None}
        return QueryRun(uuid.uuid4().hex[:12], question, chunk_set, mode, k, pool, rerank, profile, settings.rrf_k, views, hits, timings,
                        scope_report_id=report_id)

    # ---- answer ----
    def answer(self, run: QueryRun, llm_profile: str) -> AnswerRun:
        llm = self._llm_factory(llm_profile)
        shown = run.hits[:run.k]                                    # exactly what `hrag ask` would hand the model
        t0 = time.perf_counter()
        result = answer_question(run.question, shown, llm)         # THE project's answer function
        llm_ms = (time.perf_counter() - t0) * 1000
        shown_ids = [h.chunk.chunk_id for h in shown]
        # An unreadable reply is shown exactly as received and nothing is ticked: its ids were never checked, and renumbering the
        # raw text made a broken reply look like valid JSON.
        inline = [] if result.parse_error else inline_citation_ids(result.answer)
        cited = list(dict.fromkeys([*result.citations, *inline]))
        shown_text = result.answer if result.parse_error else renumber_citations(result.answer, shown_ids)
        return AnswerRun(result, shown_text, [c for c in cited if c in shown_ids],
                         [c for c in cited if c not in shown_ids], inline, round(llm_ms, 1), self.cfg.llm(llm_profile).model)

    def judge(self, run: QueryRun, ans: AnswerRun, judge_profile: str) -> dict[str, Any] | None:
        """Optional LLM check of faithfulness, using the project's own judge prompt (no key facts, so only faithfulness)."""
        judge = self._llm_factory(judge_profile)
        q = Question("playground", run.question, "lookup", True, (), (), "", False)
        return judge_answer(judge, q, ans.result, run.hits[:run.k])


# ---------- relevance marks -> metrics ----------

def record_metrics(rec: dict[str, Any], ks: Sequence[int] = KS) -> dict[str, Any]:
    """Chunk-level metrics for one saved evaluation, computed with the project's own metric functions.

    Unjudged chunks count as NOT relevant, so `judged@K` says how much of the top K you actually looked at. Recall is relative to
    the relevant chunks YOU marked inside the judged pool - the true number of relevant chunks in the corpus is unknown."""
    rows = rec["retrieved"]
    ranking = [r["chunk_id"] for r in sorted(rows, key=lambda r: r["rank"])]
    gold = {r["chunk_id"] for r in rows if r["label"] == "relevant"}
    judged = {r["chunk_id"] for r in rows if r["label"] in ("relevant", "irrelevant")}
    out: dict[str, Any] = {"n_relevant": len(gold), "n_judged": len(judged), "pool": len(ranking)}
    for k in ks:
        top = ranking[:k]
        out[f"judged@{k}"] = len(set(top) & judged) / max(len(top), 1)
        out[f"P@{k}"] = precision_at_k(ranking, gold, k) if judged else None
        out[f"R@{k}"] = recall_at_k(ranking, gold, k) if gold else None
        out[f"nDCG@{k}"] = ndcg_at_k(ranking, gold, k) if gold else None
    out["MRR"] = reciprocal_rank(ranking, gold) if judged else None
    out["latency_ms"] = rec.get("timings", {}).get("total_ms")
    return out


def config_key(rec: dict[str, Any]) -> str:
    return f"{rec['chunk_set']} | {rec['mode']}{'+rerank' if rec.get('rerank') else ''} | {rec.get('embed_profile', '')}"


def aggregate(records: Sequence[dict[str, Any]], ks: Sequence[int] = KS) -> list[dict[str, Any]]:
    """One row per configuration: mean of each metric over the saved evaluations that have it."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault(config_key(rec), []).append(record_metrics(rec, ks))
    rows = []
    for key, ms in groups.items():
        row: dict[str, Any] = {"configuration": key, "evaluations": len(ms)}
        for name in [f"P@{k}" for k in ks] + [f"R@{k}" for k in ks] + ["MRR"] + [f"nDCG@{k}" for k in ks] + ["latency_ms"]:
            row[name] = _mean([m.get(name) for m in ms])
        rows.append(row)
    return rows


def answer_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    acc: dict[str, int] = {}
    faith: dict[str, int] = {}
    for rec in records:
        a = rec.get("answer") or {}
        if a.get("accuracy") and a["accuracy"] != "not rated":
            acc[a["accuracy"]] = acc.get(a["accuracy"], 0) + 1
        if a.get("faithfulness") and a["faithfulness"] != "not rated":
            faith[a["faithfulness"]] = faith.get(a["faithfulness"], 0) + 1
    rated_a, rated_f = sum(acc.values()), sum(faith.values())
    return {"accuracy": acc, "faithfulness": faith,
            "accuracy_correct_rate": round(acc.get("correct", 0) / rated_a, 4) if rated_a else None,
            "faithful_rate": round(faith.get(FAITHFULNESS[1], 0) / rated_f, 4) if rated_f else None,
            "rated_accuracy": rated_a, "rated_faithfulness": rated_f}


# ---------- saving ----------

class JudgmentLog:
    """Append-only JSONL beside your eval files. One line = one saved evaluation. The database is never touched."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lead = ""
        if self.path.is_file() and self.path.stat().st_size:
            with self.path.open("rb") as f:
                f.seek(-1, 2)
                lead = "" if f.read(1) == b"\n" else "\n"      # a torn last line must not swallow this record
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(lead + json.dumps(record, ensure_ascii=False) + "\n")

    def load(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue                                       # a half-written line must not hide the rest
        return out


def judgment_log(cfg: Config) -> JudgmentLog:
    return JudgmentLog(cfg.eval_dir / "playground" / "judgments.jsonl")


def build_record(run: QueryRun, labels: dict[int, str], ans: AnswerRun | None, accuracy: str, faithfulness: str, notes: str,
                 judge: dict[str, Any] | None = None, llm_profile: str = "") -> dict[str, Any]:
    cited = set(ans.cited_valid) if ans else set()
    retrieved = [{"chunk_id": v.chunk_id, "report_id": v.report_id, "rank": v.rank, "score": v.score,
                  "supplied": v.supplied, "cited": v.chunk_id in cited, "label": labels.get(v.chunk_id, "unjudged"),
                  "vector_rank": v.vector_rank, "bm25_rank": v.bm25_rank, "section": v.section} for v in run.views]
    return {
        "version": SCHEMA_VERSION, "run_id": run.id, "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": run.question, "chunk_set": run.chunk_set, "mode": run.mode, "k": run.k, "pool": run.pool,
        "rerank": run.rerank, "embed_profile": run.embed_profile, "timings": run.timings, "retrieved": retrieved,
        "relevant_report_ids": sorted({r["report_id"] for r in retrieved if r["label"] == "relevant"}),
        "answer": ({"llm_profile": llm_profile, "model": ans.model, "text": ans.result.answer, "abstained": ans.result.abstained,
                    "parse_error": ans.result.parse_error, "cited": ans.cited_valid, "cited_invalid": ans.cited_invalid,
                    "llm_ms": ans.llm_ms, "accuracy": accuracy, "faithfulness": faithfulness, "judge": judge} if ans else None),
        "notes": notes,
    }


# ---------- compare configurations on the questions you already judged ----------

def compare_configs(backend: Backend, records: Sequence[dict[str, Any]], *, chunk_sets: Sequence[str], modes: Sequence[str],
                    k: int, embed_profile: str | None = None, pool: int = 20) -> list[dict[str, Any]]:
    """Re-run the SAME questions under other chunk sets / modes and score them against the REPORTS you marked relevant.
    Report level, because chunk ids differ between chunk sets. A question without any relevant mark is skipped."""
    gold_by_question: dict[str, set[int]] = {}
    for rec in records:
        if rec.get("relevant_report_ids"):
            gold_by_question.setdefault(rec["question"], set()).update(rec["relevant_report_ids"])
    if not gold_by_question:
        raise PlaygroundError("no saved evaluation has a chunk marked relevant yet - judge and save a few first")
    rows = []
    for cs in chunk_sets:
        for mode in modes:
            per = []
            for q, gold in gold_by_question.items():
                run = backend.retrieve(q, chunk_set=cs, mode=mode, k=k, pool=pool, embed_profile=embed_profile)
                ranking = report_ranking(run.hits)
                per.append({"P": precision_at_k(ranking, gold, k), "R": recall_at_k(ranking, gold, k),
                            "MRR": reciprocal_rank(ranking, gold), "nDCG": ndcg_at_k(ranking, gold, k),
                            "ms": run.timings["search_ms"]})
            rows.append({"chunk_set": cs, "mode": mode, "questions": len(per), f"P@{k}": _mean([p["P"] for p in per]),
                         f"R@{k}": _mean([p["R"] for p in per]), "MRR": _mean([p["MRR"] for p in per]),
                         f"nDCG@{k}": _mean([p["nDCG"] for p in per]), "search_ms (excl. embedding)": _mean([p["ms"] for p in per])})
    return rows


# ---------- chunk explorer helpers ----------

def chunk_checks(chunk: ChunkRow, min_tokens: int) -> list[str]:
    """Cheap warnings about how a chunk was cut. Heuristics to look at, not verdicts."""
    flags = []
    if chunk.text.count("```") % 2 == 1:
        flags.append("code fence split (odd number of ``` markers)")
    if chunk.text[:1].islower():
        flags.append("starts mid-sentence (lower-case first letter)")
    if chunk.n_tokens < max(min_tokens, 1) * 1.0:
        flags.append(f"very small ({chunk.n_tokens} est. tokens)")
    if chunk.text.rstrip()[-1:] not in (".", "!", "?", ":", "`", ")", "]", "}", "\"", "'", "*") and chunk.text.strip():
        flags.append("does not end at sentence end")
    return flags


def chunk_set_summary(chunks: Sequence[ChunkRow], min_tokens: int) -> dict[str, Any]:
    tokens = [c.n_tokens for c in chunks]
    flagged = [c for c in chunks if chunk_checks(c, min_tokens)]
    per_report: dict[int, int] = {}
    for c in chunks:
        per_report[c.report_id] = per_report.get(c.report_id, 0) + 1
    return {"chunks": len(chunks), "reports": len(per_report),
            "tokens_min": min(tokens, default=0), "tokens_median": int(statistics.median(tokens)) if tokens else 0,
            "tokens_max": max(tokens, default=0), "chunks_per_report_max": max(per_report.values(), default=0),
            "with_code": sum(c.has_code for c in chunks),
            "split_code_fence": sum(c.text.count("```") % 2 == 1 for c in chunks),
            "flagged_any": len(flagged), "flagged_share": round(len(flagged) / len(chunks), 3) if chunks else 0.0}


def overlap_preview(prev: ChunkRow, nxt: ChunkRow, width: int = 160) -> tuple[str, str]:
    return prev.text[-width:], nxt.text[:width]


# ---------- report library ----------

REPORT_URL = "https://hackerone.com/reports/{}"
CSV_COLUMNS = ("report_id", "title", "program", "weakness", "disclosed_at", "hackerone_url", "stored_text_differs_from_dataset")


def report_url(report_id: int) -> str:
    """The public HackerOne page of a report, built ONLY from its integer id - never from text stored in the database."""
    if isinstance(report_id, bool) or not isinstance(report_id, int) or report_id <= 0:
        raise PlaygroundError(f"not a report id: {report_id!r}")
    return REPORT_URL.format(report_id)


def search_reports(reports: Iterable[ReportRow], query: str) -> list[ReportRow]:
    """Reports matching EVERY word of `query` in the id, title, program or weakness (case-insensitive substring), by id."""
    terms = [t.lstrip("#") for t in query.lower().split()]
    terms = [t for t in terms if t]
    out = []
    for r in sorted(reports, key=lambda r: r.report_id):
        fields = (str(r.report_id), r.title.lower(), r.program.lower(), r.weakness.lower())
        if all(any(t in f for f in fields) for t in terms):
            out.append(r)
    return out


class ScopedRetriever(Retriever):
    """The project's own Retriever, restricted to the chunks of ONE report.

    Everything that ranks (search, Reciprocal Rank Fusion, near-duplicate suppression, the optional rerank) is inherited unchanged;
    only the corpus it ranks over is smaller. Two consequences, both shown in the page: BM25 statistics (idf, average length) are
    computed over this report's chunks only, so keyword SCORES are not comparable with the Ask tab's; cosine scores are identical.
    Reads private attributes of Retriever; tests/test_report_library.py compares it with a plain Retriever built over just this
    report, so a change in Retriever that breaks the copy fails loudly."""

    def __init__(self, parent: Retriever, report_id: int):
        ids = sorted(cid for cid, c in parent.chunks.items() if c.report_id == report_id)
        if not ids:
            raise PlaygroundError(f"report {report_id} has no chunks in chunk set '{parent.chunk_set}'")
        self.chunk_set, self.profile = parent.chunk_set, parent.profile
        self.embedder, self.reranker = parent.embedder, parent.reranker
        self.chunks = {cid: parent.chunks[cid] for cid in ids}
        self._bm25 = Bm25Index(ids, [self.chunks[i].embed_text for i in ids])
        rows = [(cid, parent._row_of[cid]) for cid in ids if cid in parent._row_of]
        self._vec_ids = [cid for cid, _ in rows]
        self._matrix = parent._matrix[[row for _, row in rows]] if rows else np.zeros((0, 0), dtype=np.float32)
        self._row_of = {cid: i for i, cid in enumerate(self._vec_ids)}


def read_original_texts(cfg: Config, wanted: set[int]) -> dict[int, str]:
    """{report id: the dataset's original text} for `wanted`, from the cached parquet files; {} when pyarrow or the files are absent."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {}
    found: dict[int, str] = {}
    for name in cfg.ingest.dataset_files:
        path = cfg.cache_dir / name
        if not path.is_file():
            continue
        table = pq.read_table(path, columns=["id", "vulnerability_information"])
        for rid, text in zip(table.column("id").to_pylist(), table.column("vulnerability_information").to_pylist()):
            if int(rid) in wanted and int(rid) not in found:
                found[int(rid)] = text or ""
    return found


def _squash(text: str) -> str:
    return " ".join(text.replace("\r\n", "\n").replace("\r", "\n").split())


def visible(text: str) -> str:
    """Make invisible and control characters readable (<U+E0041>) so a removed hidden payload can be seen in a diff."""
    return "".join(f"<U+{ord(c):04X}>" if unicodedata.category(c) in ("Cf", "Co", "Cn", "Cc") and c not in "\n\t" else c for c in text)


def cleaning_diff(original: str, stored: str, max_lines: int = 200, max_width: int = 400) -> dict[str, Any]:
    """How the stored (cleaned) text differs from the dataset's original: whether it differs at all, how much text is left,
    and the changed lines. Whitespace-only differences do not count."""
    o, s = _squash(original), _squash(stored)
    if o == s:
        return {"identical": True, "kept_share": 1.0, "original_chars": len(o), "stored_chars": len(s), "diff": ""}
    a = [ln.strip() for ln in original.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    b = [ln.strip() for ln in stored.split("\n")]
    lines = [ln for ln in difflib.unified_diff(a, b, "dataset original", "stored in SQLite", lineterm="", n=0) if not ln.startswith("@@")]
    clipped = [visible(ln)[:max_width] for ln in lines[:max_lines]]
    if len(lines) > max_lines:
        clipped.append(f"... {len(lines) - max_lines} more changed lines not shown")
    return {"identical": False, "kept_share": round(len(s) / max(len(o), 1), 3), "original_chars": len(o), "stored_chars": len(s),
            "diff": "\n".join(clipped)}


def _csv_safe(value: str) -> str:
    """Spreadsheets run a cell that starts with = + - @ as a formula, and titles are written by strangers: defuse it."""
    return "'" + value if value[:1] in ("=", "+", "-", "@", "\t", "\r") else value


def reports_csv(reports: Iterable[ReportRow], originals: dict[int, str]) -> bytes:
    """UTF-8 with a BOM (so Excel reads the titles correctly). `stored_text_differs_from_dataset` is yes / no / unknown."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n")
    writer.writerow(CSV_COLUMNS)
    for r in sorted(reports, key=lambda r: r.report_id):
        original = originals.get(r.report_id)
        differs = "unknown" if original is None else ("no" if _squash(original) == _squash(r.body) else "yes")
        writer.writerow([r.report_id, _csv_safe(r.title), _csv_safe(r.program), _csv_safe(r.weakness), r.disclosed_at[:10],
                         report_url(r.report_id), differs])
    return ("﻿" + buf.getvalue()).encode("utf-8")
