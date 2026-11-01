"""Typed access to config.toml. No secrets live here - see secrets.py."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDING_PROVIDERS = {"openai", "voyage", "openai_compat"}
LLM_PROVIDERS = {"anthropic", "openai_compat"}
RERANK_PROVIDERS = {"voyage"}
RETRIEVAL_MODES = {"vector", "bm25", "hybrid"}


class ConfigError(ValueError):
    """The config file is missing something or has a bad value."""


@dataclass(frozen=True)
class EmbeddingProfile:
    name: str
    provider: str
    model: str
    api_key_name: str
    base_url: str = ""
    dimensions: int = 0          # 0 = provider default
    query_prefix: str = ""
    doc_prefix: str = ""
    batch_size: int = 64
    timeout_s: float = 60.0
    max_retries: int = 6


@dataclass(frozen=True)
class LLMProfile:
    name: str
    provider: str
    model: str
    api_key_name: str
    base_url: str = ""
    max_tokens: int = 1500
    temperature: float | None = 0.0   # None = do not send
    max_tokens_param: str = "max_tokens"
    timeout_s: float = 120.0
    max_retries: int = 4


@dataclass(frozen=True)
class RerankProfile:
    name: str
    provider: str
    model: str
    api_key_name: str
    base_url: str = ""
    timeout_s: float = 60.0
    max_retries: int = 4


@dataclass(frozen=True)
class ChunkSetCfg:
    name: str
    max_tokens: int
    overlap_tokens: int
    min_tokens: int


@dataclass(frozen=True)
class IngestSettings:
    n_reports: int
    seed: int
    min_body_chars: int
    allowed_substates: tuple[str, ...]
    near_duplicate_jaccard: float
    dataset_base_url: str
    dataset_files: tuple[str, ...]


@dataclass(frozen=True)
class CollectSettings:
    n_reports: int = 100               # how many reports to keep
    min_body_chars: int = 400          # after cleaning; below this a report is rejected as trivial
    sparse_chars: int = 600            # below this a kept report is flagged 'sparse'
    per_program: int = 3               # at most this many reports from one HackerOne program
    per_weakness: int = 8              # at most this many with the same weakness / CWE
    max_gitlab: int = 30               # cap for GitLab-copy top-up reports (they are all one program); pinned links are exempt
    near_duplicate_jaccard: float = 0.8
    gitlab_urls_file: str = "sources/gitlab_urls.txt"
    gitlab_project: str = "gitlab-org/gitlab"


@dataclass(frozen=True)
class RetrievalSettings:
    mode: str = "hybrid"
    top_k: int = 5
    candidates: int = 30
    rrf_k: int = 60
    max_chunks_per_report: int = 2
    dedupe_cosine: float = 0.95
    rerank: bool = False


@dataclass(frozen=True)
class Defaults:
    chunk_set: str
    embedding: str
    answer: str
    judge: str
    rerank: str
    enrich: str = "openrouter_enrich"      # the LLM profile that writes 'When to try this'


@dataclass(frozen=True)
class Config:
    root: Path
    db_path: Path
    cache_dir: Path
    eval_dir: Path
    ingest: IngestSettings
    collect: CollectSettings
    retrieval: RetrievalSettings
    defaults: Defaults
    chunk_sets: dict[str, ChunkSetCfg]
    embeddings: dict[str, EmbeddingProfile]
    llms: dict[str, LLMProfile]
    rerankers: dict[str, RerankProfile]

    def chunk_set(self, name: str | None = None) -> ChunkSetCfg:
        return _lookup(self.chunk_sets, name or self.defaults.chunk_set, "chunk set")

    def embedding(self, name: str | None = None) -> EmbeddingProfile:
        return _lookup(self.embeddings, name or self.defaults.embedding, "embedding profile")

    def llm(self, name: str) -> LLMProfile:
        return _lookup(self.llms, name, "llm profile")

    def answer_llm(self, name: str | None = None) -> LLMProfile:
        return self.llm(name or self.defaults.answer)

    def judge_llm(self, name: str | None = None) -> LLMProfile:
        return self.llm(name or self.defaults.judge)

    def reranker(self, name: str | None = None) -> RerankProfile:
        return _lookup(self.rerankers, name or self.defaults.rerank, "rerank profile")


def _lookup(table: dict[str, Any], name: str, what: str) -> Any:
    if name not in table:
        raise ConfigError(f"unknown {what} '{name}'. Available: {', '.join(sorted(table)) or '(none)'}")
    return table[name]


def _need(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ConfigError(f"config.toml: [{where}] is missing '{key}'")
    return d[key]


def _check_provider(value: str, allowed: set[str], where: str) -> str:
    if value not in allowed:
        raise ConfigError(f"config.toml: [{where}] provider '{value}' must be one of {sorted(allowed)}")
    return value


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path or os.environ.get("HRAG_CONFIG") or PROJECT_ROOT / "config.toml")
    if not cfg_path.is_file():
        raise ConfigError(f"config file not found: {cfg_path}")
    raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    root = cfg_path.resolve().parent

    paths = raw.get("paths", {})

    def resolve(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else root / q

    ing = _need(raw, "ingest", "ingest")
    ingest = IngestSettings(
        n_reports=int(ing.get("n_reports", 100)),
        seed=int(ing.get("seed", 42)),
        min_body_chars=int(ing.get("min_body_chars", 400)),
        allowed_substates=tuple(ing.get("allowed_substates", ["resolved"])),
        near_duplicate_jaccard=float(ing.get("near_duplicate_jaccard", 0.8)),
        dataset_base_url=str(_need(ing, "dataset_base_url", "ingest")),
        dataset_files=tuple(_need(ing, "dataset_files", "ingest")),
    )

    c = raw.get("collect", {})
    collect = CollectSettings(
        n_reports=int(c.get("n_reports", 100)), min_body_chars=int(c.get("min_body_chars", 400)),
        sparse_chars=int(c.get("sparse_chars", 600)), per_program=int(c.get("per_program", 3)),
        per_weakness=int(c.get("per_weakness", 8)), max_gitlab=int(c.get("max_gitlab", 30)),
        near_duplicate_jaccard=float(c.get("near_duplicate_jaccard", 0.8)),
        gitlab_urls_file=str(c.get("gitlab_urls_file", "sources/gitlab_urls.txt")),
        gitlab_project=str(c.get("gitlab_project", "gitlab-org/gitlab")))

    r = raw.get("retrieval", {})
    retrieval = RetrievalSettings(
        mode=str(r.get("mode", "hybrid")),
        top_k=int(r.get("top_k", 5)),
        candidates=int(r.get("candidates", 30)),
        rrf_k=int(r.get("rrf_k", 60)),
        max_chunks_per_report=int(r.get("max_chunks_per_report", 2)),
        dedupe_cosine=float(r.get("dedupe_cosine", 0.95)),
        rerank=bool(r.get("rerank", False)),
    )
    if retrieval.mode not in RETRIEVAL_MODES:
        raise ConfigError(f"config.toml: [retrieval] mode must be one of {sorted(RETRIEVAL_MODES)}")

    dflt = _need(raw, "defaults", "defaults")
    defaults = Defaults(**{k: str(_need(dflt, k, "defaults")) for k in ("chunk_set", "embedding", "answer", "judge", "rerank")},
                        enrich=str(dflt.get("enrich", "openrouter_enrich")))

    chunk_sets = {
        name: ChunkSetCfg(name, int(_need(c, "max_tokens", f"chunk_sets.{name}")),
                          int(c.get("overlap_tokens", 0)), int(c.get("min_tokens", 0)))
        for name, c in raw.get("chunk_sets", {}).items()
    }

    embeddings = {}
    for name, e in raw.get("embeddings", {}).items():
        where = f"embeddings.{name}"
        embeddings[name] = EmbeddingProfile(
            name=name,
            provider=_check_provider(str(_need(e, "provider", where)), EMBEDDING_PROVIDERS, where),
            model=str(_need(e, "model", where)),
            api_key_name=str(_need(e, "api_key_name", where)),
            base_url=str(e.get("base_url", "")).rstrip("/"),
            dimensions=int(e.get("dimensions", 0)),
            query_prefix=str(e.get("query_prefix", "")),
            doc_prefix=str(e.get("doc_prefix", "")),
            batch_size=int(e.get("batch_size", 64)),
            timeout_s=float(e.get("timeout_s", 60.0)),
            max_retries=int(e.get("max_retries", 6)),
        )
        if embeddings[name].provider == "openai_compat" and not embeddings[name].base_url:
            raise ConfigError(f"config.toml: [{where}] needs base_url for provider openai_compat")

    llms = {}
    for name, m in raw.get("llm", {}).items():
        where = f"llm.{name}"
        temp = float(m.get("temperature", 0.0))
        llms[name] = LLMProfile(
            name=name,
            provider=_check_provider(str(_need(m, "provider", where)), LLM_PROVIDERS, where),
            model=str(_need(m, "model", where)),
            api_key_name=str(_need(m, "api_key_name", where)),
            base_url=str(m.get("base_url", "")).rstrip("/"),
            max_tokens=int(m.get("max_tokens", 1500)),
            temperature=None if temp < 0 else temp,
            max_tokens_param=str(m.get("max_tokens_param", "max_tokens")),
            timeout_s=float(m.get("timeout_s", 120.0)),
            max_retries=int(m.get("max_retries", 4)),
        )
        if llms[name].provider == "openai_compat" and not llms[name].base_url:
            raise ConfigError(f"config.toml: [{where}] needs base_url for provider openai_compat")

    rerankers = {}
    for name, m in raw.get("rerank", {}).items():
        where = f"rerank.{name}"
        rerankers[name] = RerankProfile(
            name=name,
            provider=_check_provider(str(_need(m, "provider", where)), RERANK_PROVIDERS, where),
            model=str(_need(m, "model", where)),
            api_key_name=str(_need(m, "api_key_name", where)),
            base_url=str(m.get("base_url", "")).rstrip("/"),
            timeout_s=float(m.get("timeout_s", 60.0)),
            max_retries=int(m.get("max_retries", 4)),
        )

    return Config(
        root=root,
        db_path=resolve(paths.get("db", "data/hrag.db")),
        cache_dir=resolve(paths.get("cache_dir", "data/cache")),
        eval_dir=resolve(paths.get("eval_dir", "eval")),
        ingest=ingest,
        collect=collect,
        retrieval=retrieval,
        defaults=defaults,
        chunk_sets=chunk_sets,
        embeddings=embeddings,
        llms=llms,
        rerankers=rerankers,
    )
