"""Command line: python -m hrag <command>.  See README.md for the order to run things in."""
from __future__ import annotations

import argparse
import getpass
import json
import platform
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .answer import answer_question
from .config import Config, ConfigError, RetrievalSettings, load_config
from . import collect as collect_mod
from . import enrich as enrich_mod
from . import gitlab_src, h1_api
from .embeddings import Embedder, EmbeddingError
from .evaluate import (EvalError, check_against_store, compare_rows, evaluate_answers, evaluate_retrieval, load_questions,
                       write_results)
from .ingest import IngestError, download_dataset, read_rows, select_reports, write_summary
from .llm import LLM, LLMError
from .pipeline import PipelineError, build_chunks, embed_chunks
from .questions import draft_questions, write_jsonl
from .rerank import Reranker, RerankError
from .retrieve import Retriever
from . import secrets
from .secrets import MissingKeyError, get_secret, require_secret, secret_source, set_secret
from .store import Store

KNOWN_ERRORS = (h1_api.H1Error, gitlab_src.GitLabError, ConfigError, MissingKeyError, EmbeddingError, LLMError, RerankError, PipelineError, IngestError, EvalError, ValueError)
COLLECT_NOTICE = ("Sources: reports fetched with YOUR HackerOne API token, and GitLab's public copies of disclosed HackerOne reports.\n"
                  "Treat the text as private working data: do not redistribute it, and confirm with whoever owns the policy that sending it\n"
                  "to an embedding / LLM provider is allowed.")
TERMS_NOTICE = ("Data: Hugging Face dataset Hacker0x01/hackerone_disclosed_reports, 'made available under HackerOne's Terms of Service'.\n"
                "Reuse terms beyond that were not found - keep the text for your own private use and do not redistribute it.")


# ---------- small helpers ----------

def table(rows: Sequence[Sequence[Any]]) -> str:
    cells = [[("-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(cells[0]))]
    return "\n".join("  ".join(v.ljust(w) for v, w in zip(r, widths)).rstrip() for r in cells)


def open_store(cfg: Config) -> Store:
    return Store(cfg.db_path)


def make_embedder(cfg: Config, profile_name: str | None) -> tuple[Embedder, Any]:
    profile = cfg.embedding(profile_name)
    return Embedder(profile, require_secret(profile.api_key_name)), profile


def make_llm(cfg: Config, name: str) -> LLM:
    profile = cfg.llm(name)
    return LLM(profile, require_secret(profile.api_key_name))


def make_retriever(cfg: Config, store: Store, chunk_set: str, profile_name: str | None, *, need_vectors: bool, need_rerank: bool) -> Retriever:
    embedder, profile = (make_embedder(cfg, profile_name) if need_vectors else (None, None))
    reranker = None
    if need_rerank:
        rp = cfg.reranker()
        reranker = Reranker(rp, require_secret(rp.api_key_name))
    return Retriever(store, chunk_set, profile=profile.name if profile else None, embedder=embedder, reranker=reranker)


def settings_with(cfg: Config, **overrides: Any) -> RetrievalSettings:
    data = {**cfg.retrieval.__dict__, **{k: v for k, v in overrides.items() if v is not None}}
    return RetrievalSettings(**data)


def parse_arm(arm: str) -> tuple[str, bool]:
    mode, _, suffix = arm.partition("+")
    if mode not in ("vector", "bm25", "hybrid") or suffix not in ("", "rerank"):
        raise EvalError(f"bad arm '{arm}' - use vector, bm25 or hybrid, optionally with +rerank (e.g. hybrid+rerank)")
    return mode, suffix == "rerank"


def require_reviewed(questions, allow_unreviewed: bool) -> None:
    pending = [q.id for q in questions if q.needs_review]
    if pending and not allow_unreviewed:
        raise EvalError(f"{len(pending)} question(s) are still marked needs_review (e.g. {', '.join(pending[:5])}). Read each one, fix the gold "
                        "report ids / key facts, then set needs_review to false - or pass --allow-unreviewed for a dry run.")


def common_search_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--set", dest="chunk_set", help="chunk set (default from config.toml)")
    p.add_argument("--embed", help="embedding profile (default from config.toml)")
    p.add_argument("--mode", choices=["vector", "bm25", "hybrid"], help="semantic, keyword or both (default from config.toml)")
    p.add_argument("--k", type=int, help="how many chunks to return")
    p.add_argument("--rerank", action="store_true", help="rerank candidates with the configured reranker")


# ---------- commands ----------

def cmd_doctor(cfg: Config, args) -> int:
    ok = sys.version_info >= (3, 11)
    print(f"python      {platform.python_version()}  {'ok' if ok else 'TOO OLD - need 3.11 or newer'}")
    print(f"platform    {platform.platform()}")
    print(f"sqlite      {sqlite3.sqlite_version}")
    for mod in ("numpy", "httpx", "pyarrow", "keyring"):
        try:
            m = __import__(mod)
            print(f"{mod:<11} {getattr(m, '__version__', 'ok')}")
        except ImportError:
            print(f"{mod:<11} MISSING - run setup.bat (or pip install -r requirements.txt)")
            ok = False
    print(f"config      {cfg.root / 'config.toml'}")
    print(f"database    {cfg.db_path}  ({'exists' if cfg.db_path.exists() else 'not created yet'})")
    names: dict[str, list[str]] = {}
    for kind, profiles in (("embedding", cfg.embeddings), ("llm", cfg.llms), ("rerank", cfg.rerankers)):
        for p in profiles.values():
            names.setdefault(p.api_key_name, []).append(f"{kind}:{p.name}")
    names.setdefault("H1_API_USERNAME", []).append("collector")
    names.setdefault("H1_API_TOKEN", []).append("collector")
    names.setdefault("GITLAB_TOKEN", []).append("collector (optional: only needed to read GitLab comments)")
    dotenv = secrets._dotenv_path()
    print(f".env file   {dotenv}  ({'found' if dotenv.is_file() else 'not found - copy .env.example to .env and fill it in'})")
    print("\nAPI keys (values are never shown; looked up in: environment variable, .env file, credential store):")
    for key, used_by in sorted(names.items()):
        print(f"  {key:<20} {secret_source(key):<8} used by {', '.join(used_by)}")
    return 0 if ok else 1


def cmd_keys(cfg: Config, args) -> int:
    if args.keys_cmd == "set":
        value = getpass.getpass(f"Paste the value for {args.name} (input is hidden): ").strip()
        if not value:
            print("nothing entered - nothing stored")
            return 1
        set_secret(args.name, value)
        print(f"stored {args.name} in the OS credential store")
        return 0
    return cmd_doctor(cfg, args)


# ---------- HackerOne API collector ----------

def make_h1_client(cfg: Config) -> h1_api.H1Client:
    return h1_api.H1Client(require_secret("H1_API_USERNAME"), require_secret("H1_API_TOKEN"), cache_dir=cfg.cache_dir / "h1")


def cmd_h1(cfg: Config, args) -> int:
    probe_path = cfg.db_path.parent / "h1_probe.json"
    candidates_path = cfg.db_path.parent / "h1_candidates.jsonl"
    if args.h1_cmd == "probe":
        verdict = h1_api.run_probe(make_h1_client(cfg), query=args.query)
        h1_api.write_probe(probe_path, verdict)
        print(json.dumps({k: v for k, v in verdict.items() if k not in ("item_shape", "detail_shape")}, indent=2))
        print("\nitem shape (field names only):\n" + json.dumps(verdict["item_shape"], indent=2))
        print("\ndetail shape (field names only):\n" + json.dumps(verdict["detail_shape"], indent=2))
        print(f"\nsaved {probe_path}")
        print("RESULT: " + ("API returns full report text - bulk collection is now allowed." if verdict["usable_for_bulk"]
                            else "NOT usable for bulk collection yet - see notes above. Nothing was collected."))
        return 0 if verdict["usable_for_bulk"] else 1
    h1_api.require_probe_ok(probe_path)
    if args.h1_cmd == "discover":
        client = make_h1_client(cfg)
        pool = list(client.iter_hacktivity(args.query, max_items=args.scan))
        picked = h1_api.pick_diverse(pool, args.n, args.per_program)
        candidates_path.parent.mkdir(parents=True, exist_ok=True)
        with candidates_path.open("w", encoding="utf-8") as fh:
            for it in picked:
                fh.write(json.dumps({"report_id": h1_api.report_id_of(it), "program": h1_api.program_of(it),
                                     "title": h1_api.title_of(it)}, ensure_ascii=False) + "\n")
        programs = {h1_api.program_of(it) or "(unknown)" for it in picked}
        print(f"scanned {len(pool)} disclosed items, picked {len(picked)} reports from {len(programs)} programs -> {candidates_path}")
        return 0 if picked else 1
    if not candidates_path.is_file():
        raise h1_api.H1Error("no candidate list yet - run  python -m hrag h1 discover  first")
    ids = [json.loads(line)["report_id"] for line in candidates_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    client = make_h1_client(cfg)
    ok, failed = 0, []
    for rid in ids:
        try:
            client.get_report(rid)
            ok += 1
        except h1_api.H1Error as exc:
            failed.append((rid, exc.status))
    print(f"fetched {ok} of {len(ids)} reports into {cfg.cache_dir / 'h1'}" + (f"; failed: {failed}" if failed else ""))
    return 0 if ok else 1


# ---------- GitLab copies, collect, enrich ----------

def make_gitlab_client(cfg: Config) -> gitlab_src.GitLabClient:
    return gitlab_src.GitLabClient(cache_dir=cfg.cache_dir / "gitlab", token=get_secret("GITLAB_TOKEN"))


def make_enrich_llm(cfg: Config, name: str | None, require: bool):
    """The LLM that writes 'When to try this', or None (template text) when no usable model/key exists and it is not required."""
    profile = cfg.llm(name or cfg.defaults.enrich)
    try:
        return LLM(profile, require_secret(profile.api_key_name))
    except (MissingKeyError, LLMError) as exc:
        if require:
            raise
        print(f"warning: {exc}\n         -> 'When to try this' will be built from a template instead (weaker for retrieval)")
        return None


def cmd_gitlab(cfg: Config, args) -> int:
    client = make_gitlab_client(cfg)
    urls_file = cfg.root / cfg.collect.gitlab_urls_file
    if args.gitlab_cmd == "discover":
        pool = list(client.iter_hackerone_issues(cfg.collect.gitlab_project, max_items=args.scan))
        picked = gitlab_src.pick_candidates(pool, args.n, args.per_weakness, min_chars=cfg.collect.min_body_chars)
        for issue in picked:
            client.cache_issue(cfg.collect.gitlab_project, issue)
        print(f"looked through {len(pool)} public HackerOne-labelled issues, picked {len(picked)} complete ones (max {args.per_weakness} per weakness)\n"
              f"cached in {cfg.cache_dir / 'gitlab'} - they are all from ONE program, so `collect` caps them at {cfg.collect.max_gitlab}")
        return 0 if picked else 1
    urls = collect_mod.read_urls(urls_file)
    if not urls:
        raise gitlab_src.GitLabError(f"no links found - put your sample links (one per line) in {urls_file}")
    ok, failed = 0, []
    for url in urls:
        try:
            client.fetch_report(url)
            ok += 1
        except gitlab_src.GitLabError as exc:
            failed.append((url, str(exc)))
    print(f"fetched {ok} of {len(urls)} sample links into {cfg.cache_dir / 'gitlab'}" + ("" if get_secret("GITLAB_TOKEN") else
          "  (no GITLAB_TOKEN: GitLab comments are skipped)"))
    for url, why in failed:
        print(f"  FAILED {url}: {why}")
    return 0 if ok else 1


def cmd_collect(cfg: Config, args) -> int:
    store = open_store(cfg)
    if store.report_ids() and not args.force:
        print("The database already has reports. Collecting again REPLACES them and deletes all chunks and embeddings "
              "(paid API work). Pass --force if you really want that.")
        return 1
    sources, pinned, notes = collect_mod.load_sources(cfg.cache_dir, cfg.root / cfg.collect.gitlab_urls_file)
    if not sources:
        raise ValueError("nothing has been downloaded yet. Run `h1 probe`, then `h1 discover` and `h1 fetch` (HackerOne API), and/or "
                         "`gitlab fetch` (your sample links) first.\n" + "\n".join(notes))
    settings = cfg.collect if args.max_gitlab is None else replace(cfg.collect, max_gitlab=args.max_gitlab)
    print(COLLECT_NOTICE + "\n")
    rows, summary = collect_mod.collect_reports(sources, settings, pinned=pinned, n=args.n)
    summary["notes"] = notes + summary["notes"]
    store.replace_reports(rows)
    path = cfg.db_path.parent / "collect_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"selected {summary['selected']} of {summary['requested']} requested reports ({summary['pinned_selected']} pinned sample links) "
          f"from {len(summary['by_program'])} program{'s' if len(summary['by_program']) != 1 else ''}")
    print("by source:", summary["by_source"], "| rejected:", summary["rejected"] or "none", "| skipped by caps:", summary["skipped"] or "none")
    print("quality flags:", summary["flags"] or "none")
    print("AI-directed instructions removed:", summary["injection"])
    if summary["shortfall"]:
        print(f"SHORTFALL: {summary['shortfall']} short of the target - fetch more reports or raise the caps in config.toml [collect].")
    for note in summary["notes"]:
        print("note:", note)
    print(f"summary -> {path}\nNext: python -m hrag enrich && python -m hrag chunk && python -m hrag embed")
    return 0


def cmd_enrich(cfg: Config, args) -> int:
    store = open_store(cfg)
    if not store.report_ids():
        raise ValueError("no reports in the database - run `python -m hrag collect` first")
    llm = make_enrich_llm(cfg, args.llm, args.require_llm)
    out = enrich_mod.enrich_store(store, llm, force=args.force, progress=lambda d, t: print(f"  {d}/{t}", flush=True))
    print(f"'When to try this': {out['reports']} reports, {out['already_done']} already done, llm={out['llm']}, template={out['template']}")
    if out["template"]:
        print("The re-run is safe: use --force after you fix the model/key to regenerate the template ones.")
    return 0


def cmd_ingest(cfg: Config, args) -> int:
    store = open_store(cfg)
    if store.report_ids() and not args.force:
        print("The database already has reports. Re-ingesting REPLACES them and deletes all chunks and embeddings "
              "(paid API work). Pass --force if you really want that.")
        return 1
    print(TERMS_NOTICE + "\n")
    if args.parquet_dir:
        paths = [Path(args.parquet_dir) / f for f in cfg.ingest.dataset_files]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise IngestError(f"missing file(s) in --parquet-dir: {', '.join(p.name for p in missing)}")
    else:
        paths = download_dataset(cfg.ingest, cfg.cache_dir)
    rows = read_rows(paths)
    reports, summary = select_reports(rows, cfg.ingest, n=args.n, seed=args.seed)
    store.replace_reports(reports)
    ids_path, json_path = write_summary(summary, cfg.db_path.parent)
    print(f"dataset rows {summary['dataset_rows']}, eligible {summary['eligible_after_filters']}, selected {summary['selected']} "
          f"(seed {summary['seed']})")
    print("removed:", ", ".join(f"{k}={v}" for k, v in summary["rejected"].items()) or "nothing")
    print(f"ids -> {ids_path}\nsummary -> {json_path}\nNext: python -m hrag chunk && python -m hrag embed")
    return 0


def cmd_chunk(cfg: Config, args) -> int:
    store = open_store(cfg)
    chunk_cfg = cfg.chunk_set(args.chunk_set)
    out = build_chunks(store, chunk_cfg)
    print(f"chunk set '{out['chunk_set']}': {out['chunks']} chunks from {out['reports']} reports "
          f"(est. tokens min/median/max {out['tokens_min']}/{out['tokens_median']}/{out['tokens_max']}, {out['with_code']} with code)")
    return 0


def cmd_embed(cfg: Config, args) -> int:
    store = open_store(cfg)
    embedder, profile = make_embedder(cfg, args.embed)
    chunk_set = cfg.chunk_set(args.chunk_set).name
    out = embed_chunks(store, chunk_set, profile, embedder, force=args.force,
                       progress=lambda done, total: print(f"  embedded {done}/{total}", flush=True))
    print(f"profile '{out['profile']}' ({out['model']}): embedded {out['embedded_now']} new, {out['already_done']} already done, {out['total']} total")
    return 0


def cmd_build(cfg: Config, args) -> int:
    """collect (if the database is empty) -> enrich -> chunk -> embed."""
    store = open_store(cfg)
    if not store.report_ids() or args.refresh_data:
        rc = cmd_collect(cfg, argparse.Namespace(n=None, force=True, max_gitlab=None))
        if rc:
            return rc
    rc = cmd_enrich(cfg, argparse.Namespace(llm=None, force=False, require_llm=False))
    if rc:
        return rc
    cmd_chunk(cfg, argparse.Namespace(chunk_set=args.chunk_set))
    return cmd_embed(cfg, argparse.Namespace(chunk_set=args.chunk_set, embed=args.embed, force=False))


def cmd_stats(cfg: Config, args) -> int:
    print(json.dumps(open_store(cfg).stats(), indent=2))
    return 0


def _retrieve(cfg: Config, args, query: str):
    store = open_store(cfg)
    chunk_set = cfg.chunk_set(args.chunk_set).name
    settings = settings_with(cfg, mode=args.mode, top_k=args.k, rerank=True if args.rerank else None)
    need_vectors = settings.mode in ("vector", "hybrid")
    retriever = make_retriever(cfg, store, chunk_set, args.embed, need_vectors=need_vectors, need_rerank=settings.rerank)
    return store, retriever, settings, retriever.search(query, settings)


def cmd_search(cfg: Config, args) -> int:
    _, _, settings, hits = _retrieve(cfg, args, args.query)
    for h in hits:
        c = h.chunk
        print(f"#{h.rank}  score {h.score:.4f}  C{c.chunk_id}  report {c.report_id}  [{c.section}]  via {h.via}")
        print(f"    {c.header}")
        if args.show_text:
            print("    " + c.text.replace("\n", "\n    "))
    if not hits:
        print("(no results)")
    return 0


def cmd_ask(cfg: Config, args) -> int:
    store, _, settings, hits = _retrieve(cfg, args, args.question)
    result = answer_question(args.question, hits, make_llm(cfg, args.answer or cfg.defaults.answer))
    print(result.answer + "\n")
    if result.parse_error:
        print("(warning: the model did not return the requested JSON, so citations could not be checked)")
    if result.abstained:
        print("The model abstained: the retrieved sources did not contain the answer.")
    by_id = {h.chunk.chunk_id: h.chunk for h in hits}
    for cid in result.citations:
        c = by_id.get(cid)
        if c is None:
            print(f"  [C{cid}]  NOT among the retrieved sources - treat as invalid")
            continue
        r = store.report(c.report_id)
        link = (r.source_url if r and r.source_url else f"https://hackerone.com/reports/{c.report_id}")
        print(f"  [C{cid}]  report #{c.report_id}  {r.title if r else ''}  {link}  ({c.section})")
    return 0


def cmd_eval(cfg: Config, args) -> int:
    store = open_store(cfg)
    chunk_set = cfg.chunk_set(getattr(args, "chunk_set", None)).name
    qfile = Path(getattr(args, "file", None) or cfg.eval_dir / "questions.jsonl")
    out_dir = cfg.eval_dir / "results"

    if args.eval_cmd == "draft":
        llm = make_llm(cfg, args.llm or cfg.defaults.answer)
        items = draft_questions(store, chunk_set, llm, n_answerable=args.n, n_unanswerable=args.unanswerable, seed=args.seed,
                                model_name=llm.profile.model)
        out = Path(args.out or cfg.eval_dir / "questions.draft.jsonl")
        write_jsonl(out, items)
        print(f"wrote {len(items)} drafted questions to {out}\nReview EVERY one (question, gold_report_ids, key_facts), "
              "set needs_review to false, add your own, then save as eval/questions.jsonl.")
        return 0

    if args.eval_cmd == "compare":
        a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (args.a, args.b))
        res = compare_rows(a["rows"], b["rows"], args.metric)
        print(f"{args.metric}: A - B = {res['mean_diff']:+.4f}  95% CI [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]  "
              f"(A better on {res['a_better']}, B better on {res['b_better']}, tied {res['ties']}; n={res['n']})  "
              f"{'DIFFERENCE LOOKS REAL' if res['significant'] else 'within noise'}")
        return 0

    questions = load_questions(qfile)
    problems = check_against_store(questions, store)
    if problems:
        raise EvalError("\n".join(problems))
    if args.eval_cmd == "validate":
        pending = sum(q.needs_review for q in questions)
        print(f"{len(questions)} questions OK ({sum(q.answerable for q in questions)} answerable, "
              f"{sum(not q.answerable for q in questions)} unanswerable); {pending} still need review")
        return 0
    require_reviewed(questions, args.allow_unreviewed)

    if args.eval_cmd == "retrieval":
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        results: dict[str, list[dict]] = {}
        summary_rows = [["arm", "n", "recall@1", "recall@3", "recall@5", "MRR", "nDCG@10", "dup@5", "ms"]]
        for arm in arms:
            mode, rerank = parse_arm(arm)
            retriever = make_retriever(cfg, store, chunk_set, args.embed, need_vectors=mode != "bm25", need_rerank=rerank)
            rows, summary = evaluate_retrieval(questions, retriever, settings_with(cfg), mode=mode, rerank=rerank)
            results[arm] = rows
            m = summary["mean"]
            summary_rows.append([arm, summary["n_answerable"], m.get("recall@1"), m.get("recall@3"), m.get("recall@5"), m.get("rr"),
                                 m.get("ndcg@10"), m.get("dup_rate@5"), summary["mean_latency_ms"]])
            meta = {"kind": "retrieval", "arm": arm, "chunk_set": chunk_set, "embedding": cfg.embedding(args.embed).__dict__ if mode != "bm25" else None,
                    "retrieval": settings_with(cfg).__dict__, "questions": str(qfile), "n_questions": len(questions), "version": __version__}
            csv_path, _ = write_results(out_dir, f"retrieval-{arm.replace('+', '-')}", rows, summary, meta)
            print(f"  {arm}: per-question rows -> {csv_path}")
        print("\n" + table(summary_rows))
        if len(arms) > 1:
            print(f"\nPaired comparison on reciprocal rank vs '{arms[0]}':")
            for arm in arms[1:]:
                r = compare_rows(results[arm], results[arms[0]], "rr")
                print(f"  {arm:<16} {r['mean_diff']:+.3f}  CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  "
                      f"better on {r['a_better']}, worse on {r['b_better']}  {'REAL' if r['significant'] else 'within noise'}")
        return 0

    if args.eval_cmd == "answers":
        mode, rerank = parse_arm(args.arm)
        answer_profile, judge_profile = cfg.answer_llm(args.answer), cfg.judge_llm(args.judge)
        if (answer_profile.provider, answer_profile.base_url) == (judge_profile.provider, judge_profile.base_url):
            print("WARNING: the answer model and the judge come from the same provider - judges favour their own family's output. "
                  "Prefer a different company for --judge.")
        retriever = make_retriever(cfg, store, chunk_set, args.embed, need_vectors=mode != "bm25", need_rerank=rerank)
        chosen = questions[:args.limit] if args.limit else questions
        rows, summary = evaluate_answers(chosen, retriever, settings_with(cfg), make_llm(cfg, answer_profile.name),
                                         make_llm(cfg, judge_profile.name), mode=mode, rerank=rerank)
        meta = {"kind": "answers", "arm": args.arm, "chunk_set": chunk_set, "answer_model": answer_profile.model,
                "judge_model": judge_profile.model, "retrieval": settings_with(cfg).__dict__, "questions": str(qfile), "version": __version__}
        csv_path, json_path = write_results(out_dir, f"answers-{args.arm.replace('+', '-')}", rows, summary, meta)
        print(json.dumps(summary, indent=2))
        print(f"\nper-question rows -> {csv_path}\nfull results   -> {json_path}")
        return 0
    return 2


# ---------- parser ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hrag", description="Local RAG over public HackerOne reports.")
    p.add_argument("--config", help="path to config.toml (default: next to the project)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check Python, packages, config and which API keys are available")
    k = sub.add_parser("keys", help="store or list API keys (OS credential store)")
    ks = k.add_subparsers(dest="keys_cmd", required=True)
    ks.add_parser("status", help="show which keys are available")
    kset = ks.add_parser("set", help="store a key (you are prompted; input hidden)")
    kset.add_argument("name", help="e.g. OPENAI_API_KEY")

    gl = sub.add_parser("gitlab", help="GitLab's public copies of HackerOne reports")
    gls = gl.add_subparsers(dest="gitlab_cmd", required=True)
    gd = gls.add_parser("discover", help="find complete public HackerOne copies to top up the collection (all one program, capped)")
    gd.add_argument("--n", type=int, default=30)
    gd.add_argument("--per-weakness", type=int, default=3)
    gd.add_argument("--scan", type=int, default=300, help="how many listed issues to look through")
    gls.add_parser("fetch", help="download your sample links from sources/gitlab_urls.txt (cached on disk)")

    co = sub.add_parser("collect", help="clean, flag, dedupe and select the reports from what was downloaded (offline, free)")
    co.add_argument("--n", type=int, help="how many reports to keep (default from config.toml [collect])")
    co.add_argument("--max-gitlab", type=int, help="cap on GitLab top-up reports (0 = only your pinned sample links)")
    co.add_argument("--force", action="store_true", help="replace existing reports (deletes chunks and embeddings)")

    en = sub.add_parser("enrich", help="write the 'When to try this' text for every report (LLM, or a template without a key)")
    en.add_argument("--llm", help="llm profile (default from config.toml [defaults] enrich)")
    en.add_argument("--force", action="store_true", help="regenerate even for reports that already have it")
    en.add_argument("--require-llm", action="store_true", help="fail instead of falling back to the template")

    i = sub.add_parser("ingest-dataset", help="LEGACY: download the Hugging Face dataset and select reports from it")
    i.add_argument("--n", type=int)
    i.add_argument("--seed", type=int)
    i.add_argument("--parquet-dir", help="use already-downloaded parquet files from this folder")
    i.add_argument("--force", action="store_true", help="replace existing reports (deletes chunks and embeddings)")

    h = sub.add_parser("h1", help="HackerOne API collector: probe first; discover/fetch are blocked until the probe passes")
    hs = h.add_subparsers(dest="h1_cmd", required=True)
    hp = hs.add_parser("probe", help="test your token: does the API return complete report text? (prints names and lengths only)")
    hp.add_argument("--query", default="disclosed:true")
    hd = hs.add_parser("discover", help="list public disclosed reports across different programs")
    hd.add_argument("--n", type=int, default=94)
    hd.add_argument("--query", default="disclosed:true")
    hd.add_argument("--per-program", type=int, default=3, help="at most this many reports from one program")
    hd.add_argument("--scan", type=int, default=600, help="how many hacktivity items to look through")
    hs.add_parser("fetch", help="download the full report for every discovered candidate (cached on disk)")

    c = sub.add_parser("chunk", help="split reports into chunks")
    c.add_argument("--set", dest="chunk_set")
    e = sub.add_parser("embed", help="embed chunks with a hosted model")
    e.add_argument("--set", dest="chunk_set")
    e.add_argument("--embed")
    e.add_argument("--force", action="store_true", help="discard this profile's vectors and re-embed")

    b = sub.add_parser("build", help="collect (if empty) + enrich + chunk + embed with the defaults")
    b.add_argument("--set", dest="chunk_set")
    b.add_argument("--embed")
    b.add_argument("--refresh-data", action="store_true", help="re-collect (replaces reports, chunks and embeddings)")

    sub.add_parser("stats", help="what is in the database")

    s = sub.add_parser("search", help="retrieve chunks for a query")
    s.add_argument("query")
    common_search_args(s)
    s.add_argument("--show-text", action="store_true")
    a = sub.add_parser("ask", help="retrieve, then answer with citations")
    a.add_argument("question")
    common_search_args(a)
    a.add_argument("--answer", help="llm profile for the answer (default from config.toml)")

    ev = sub.add_parser("eval", help="evaluation")
    evs = ev.add_subparsers(dest="eval_cmd", required=True)
    d = evs.add_parser("draft", help="have an LLM draft questions for you to review")
    d.add_argument("--set", dest="chunk_set")
    d.add_argument("--n", type=int, default=40, help="answerable questions to draft")
    d.add_argument("--unanswerable", type=int, default=10)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--llm")
    d.add_argument("--out")
    v = evs.add_parser("validate", help="check the questions file")
    v.add_argument("--file")
    r = evs.add_parser("retrieval", help="Recall@k / MRR / nDCG for several search arms")
    r.add_argument("--file")
    r.add_argument("--set", dest="chunk_set")
    r.add_argument("--embed")
    r.add_argument("--arms", default="vector,bm25,hybrid", help="comma list: vector, bm25, hybrid, with optional +rerank")
    r.add_argument("--allow-unreviewed", action="store_true")
    an = evs.add_parser("answers", help="answer accuracy, citations, abstention")
    an.add_argument("--file")
    an.add_argument("--set", dest="chunk_set")
    an.add_argument("--embed")
    an.add_argument("--arm", default="hybrid")
    an.add_argument("--answer")
    an.add_argument("--judge")
    an.add_argument("--limit", type=int)
    an.add_argument("--allow-unreviewed", action="store_true")
    cp = evs.add_parser("compare", help="paired comparison of two saved result .json files")
    cp.add_argument("a")
    cp.add_argument("b")
    cp.add_argument("--metric", default="rr")
    return p


COMMANDS = {"doctor": cmd_doctor, "keys": cmd_keys, "gitlab": cmd_gitlab, "collect": cmd_collect, "enrich": cmd_enrich, "ingest-dataset": cmd_ingest, "chunk": cmd_chunk, "embed": cmd_embed, "build": cmd_build,
            "h1": cmd_h1, "stats": cmd_stats, "search": cmd_search, "ask": cmd_ask, "eval": cmd_eval}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        return COMMANDS[args.cmd](cfg, args)
    except KNOWN_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
