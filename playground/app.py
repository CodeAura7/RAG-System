"""RAG Playground - a local Streamlit page over the project's REAL retrieval, answer and evaluation code.

Run:  playground.bat      (or:  .venv\\Scripts\\python -m streamlit run playground\\app.py)
Everything runs on your laptop; the page only reads hrag.db and only writes eval/playground/judgments.jsonl.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from playground import backend as pb  # noqa: E402

st.set_page_config(page_title="HackerOne RAG Playground", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner=False)
def get_backend(config_path: str) -> pb.Backend:
    return pb.Backend.from_config(config_path or None)


def fmt(x, nd=4):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def df(rows):
    return pd.DataFrame(rows)


# ================= sidebar: settings =================
with st.sidebar:
    st.title("🔎 RAG Playground")
    config_path = st.text_input("Config file", "", placeholder="blank = the project's config.toml",
                                help="Same config.toml the command line uses: database, models, retrieval settings.")
    try:
        be = get_backend(config_path.strip())
        cfg = be.cfg
        info = be.overview()
    except Exception as exc:  # noqa: BLE001 - show any start-up problem instead of a stack trace
        st.error(f"Could not start: {exc}")
        st.stop()

    chunk_sets = sorted(info.get("chunks_per_set", {}))
    st.caption(f"Database: `{cfg.db_path}`  \n{info['reports']} reports · sources {info.get('sources')}")
    if not chunk_sets:
        st.error("The database has no chunks yet. Run:  hrag.bat build")
        st.stop()

    default_cs = chunk_sets.index(cfg.defaults.chunk_set) if cfg.defaults.chunk_set in chunk_sets else 0
    chunk_set = st.selectbox("Chunk set", chunk_sets, index=default_cs, format_func=lambda n: f"{n}  ({info['chunks_per_set'][n]} chunks)")
    embed_names = list(cfg.embeddings)
    embed_profile = st.selectbox("Embedding profile", embed_names, index=embed_names.index(cfg.defaults.embedding))
    n_chunks = info["chunks_per_set"][chunk_set]
    n_emb = be.embedded_count(chunk_set, embed_profile)
    vectors_ready = n_emb == n_chunks
    modes = list(pb.MODES) if vectors_ready else ["bm25"]
    if not vectors_ready:
        st.warning(f"Vector/hybrid search is off: {n_emb} of {n_chunks} chunks are embedded for '{embed_profile}'. Run:  hrag.bat embed"
                   + (f" --set {chunk_set}" if chunk_set != cfg.defaults.chunk_set else ""))
    mode = st.selectbox("Retrieval mode", modes, help="hybrid = vector + BM25 merged by Reciprocal Rank Fusion (the project default)")
    k = st.number_input("K = chunks given to the answer model", 1, 20, int(cfg.retrieval.top_k))
    pool = st.number_input("Judging depth (chunks shown for marking)", int(k), 30, max(int(k), 15),
                           help="Chunks ranked K+1 .. depth are shown as 'retrieved but NOT given to the LLM', so you can judge what was missed.")
    rerank = st.checkbox("Rerank", value=False, disabled=not pb.key_available(cfg.reranker().api_key_name),
                         help="Needs the reranker key from config.toml (VOYAGE_API_KEY by default).")

    llm_names = [n for n, p in cfg.llms.items() if p.model != "SET_ME"]
    usable = [n for n in llm_names if pb.key_available(cfg.llms[n].api_key_name)]
    st.divider()
    if not usable:
        st.info("No answer model has an API key yet, so you can retrieve but not generate answers. "
                "Set a key in .env (see the notes in PLAYGROUND.md).")
    answer_llm = st.selectbox("Answer model", usable or llm_names,
                              index=(usable or llm_names).index(cfg.defaults.answer) if cfg.defaults.answer in (usable or llm_names) else 0,
                              format_func=lambda n: f"{n} - {cfg.llms[n].model}" + ("" if n in usable else "  (no key)"))
    judge_llm = st.selectbox("Judge model (optional)", usable or llm_names,
                             index=(usable or llm_names).index(cfg.defaults.judge) if cfg.defaults.judge in (usable or llm_names) else 0,
                             format_func=lambda n: f"{n} - {cfg.llms[n].model}" + ("" if n in usable else "  (no key)"),
                             help="Models grade their own output generously; pick a different company from the answer model.")
    can_answer = answer_llm in usable
    if cfg.llms[answer_llm].max_tokens < 800:
        st.warning(f"'{answer_llm}' allows only {cfg.llms[answer_llm].max_tokens} output tokens. The answer is a JSON object, so it may be cut "
                   "off and show as a parse error. Use a profile with max_tokens of 1500 (see PLAYGROUND.md).")
    log = pb.judgment_log(cfg)
    st.caption(f"Marks are saved to `{log.path}`")

tab_ask, tab_chunks, tab_eval, tab_lib = st.tabs(["💬 Ask", "🧩 Chunk Explorer", "📊 Evaluation", "📚 Report Library"])


# ================= Ask =================
def render_hit(v: pb.ChunkView, run: pb.QueryRun, cited: set[int]) -> None:
    if v.chunk_id in cited:
        badge = "✅ cited"
    elif v.supplied:
        badge = "📨 given to LLM"
    else:
        badge = "🔎 retrieved only"
    with st.container(border=True):
        left, right = st.columns([3, 2])
        with left:
            n = f"[{v.rank}] " if v.supplied else ""
            st.markdown(f"**{n}#{v.rank} · {badge}**  \nC{v.chunk_id} · report [{v.report_id}]({v.source_url}) · *{v.section}* · {v.n_tokens} est. tokens  \n"
                        f"{pb.safe_markdown(v.title)}  \n`{v.program}` · {v.weakness or 'no weakness'}")
        with right:
            st.radio("Is this chunk relevant to the question?", pb.LABELS, horizontal=True, key=f"rel_{run.id}_{v.chunk_id}",
                     help="Your judgement of the CHUNK (not the report). Saved with the evaluation.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final score" + (" (RRF)" if run.mode == "hybrid" else ""), fmt(v.score, 5))
        c2.metric("Vector rank", fmt(v.vector_rank), help=None if v.vector_score is None else f"cosine similarity {v.vector_score:.4f}")
        c3.metric("BM25 rank", fmt(v.bm25_rank), help=None if v.bm25_score is None else f"BM25 score {v.bm25_score:.3f}")
        c4.metric("Vector score" if v.vector_score is not None else "BM25 score", fmt(v.vector_score if v.vector_score is not None else v.bm25_score, 3))
        with st.expander("Why it was retrieved, and the complete chunk text"):
            if run.mode == "hybrid":
                st.write(f"Reciprocal Rank Fusion with k={run.rrf_k}: score = 1/(k+vector rank) + 1/(k+BM25 rank) = "
                         f"{fmt(v.rrf_vector, 5)} + {fmt(v.rrf_bm25, 5)} = {fmt(v.score, 5)}"
                         + ("  (a chunk found by only one leg gets only that leg's term)" if not (v.vector_rank and v.bm25_rank) else ""))
            elif run.mode == "vector":
                st.write(f"Vector mode: ranked by cosine similarity {fmt(v.vector_score, 4)} between the question and the chunk text with its header.")
            else:
                st.write(f"BM25 mode: ranked by keyword score {fmt(v.bm25_score, 3)}.")
            st.caption("The label that is embedded and keyword-indexed with every chunk:")
            st.text(v.header)
            st.caption("Chunk text (shown exactly as stored; never rendered as HTML or markdown):")
            st.text(v.text)


with tab_ask:
    st.subheader("Ask a question about the reports")
    question = st.text_area("Question", key="question", height=80,
                            placeholder="e.g. How was SSRF used to reach internal services? / What does CVE-2021-44228 allow?")
    b1, b2, _ = st.columns([1, 1, 4])
    go_retrieve = b1.button("Retrieve only", help="No answer model, no LLM cost")
    go_ask = b2.button("Retrieve + answer", type="primary", disabled=not can_answer,
                       help=None if can_answer else "The selected answer model has no API key")
    if go_retrieve or go_ask:
        try:
            with st.spinner("Retrieving..."):
                run = be.retrieve(question, chunk_set=chunk_set, mode=mode, k=int(k), pool=int(pool), rerank=rerank, embed_profile=embed_profile)
            st.session_state.update(run=run, ans=None, judge=None, saved=None, ans_profile=answer_llm)
            if go_ask:
                with st.spinner(f"Asking {cfg.llms[answer_llm].model}..."):
                    st.session_state["ans"] = be.answer(run, answer_llm)
        except Exception as exc:  # noqa: BLE001
            st.error(f"{type(exc).__name__}: {exc}")

    run: pb.QueryRun | None = st.session_state.get("run")
    ans: pb.AnswerRun | None = st.session_state.get("ans")
    if run is None:
        st.info("Type a question and press a button. Retrieval runs your real hybrid search; nothing is simulated.")
    else:
        cited = set(ans.cited_valid) if ans else set()
        t = run.timings
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Query embedding", f"{t['embed_ms']:.0f} ms" if not t["embed_cached"] else "cached",
                  help=f"first call took {t['embed_first_ms']} ms" if t.get("embed_first_ms") else None)
        m2.metric("Search (BM25 + vectors + fusion)", f"{t['search_ms']:.0f} ms")
        m3.metric("Retrieval total", f"{t['total_ms']:.0f} ms")
        m4.metric("Answer model", f"{ans.llm_ms:.0f} ms" if ans else "not run")

        if ans:
            st.markdown("### Answer")
            if ans.result.abstained:
                st.warning("The model abstained: it says the retrieved sources do not contain the answer.")
            notice = pb.answer_notice(ans.result)
            if notice:
                (st.warning if notice[0] == "warning" else st.info)(notice[1])
            if ans.result.parse_error:
                st.code(ans.display_text, language=None)             # the raw reply, exactly as received (never rendered as markdown)
            else:
                st.markdown(pb.safe_markdown(ans.display_text))
            if ans.cited_invalid:
                st.error("Cited chunks that were NOT given to the model (treat as invented): " + ", ".join(f"C{c}" for c in ans.cited_invalid))
            if ans.result.citations and not ans.inline_ids:
                st.caption("The model listed citations but wrote no [..] markers in the answer text.")
            st.markdown("**Sources** (numbers match the [n] in the answer)")
            for v in run.supplied:
                mark = "✅ cited" if v.chunk_id in cited else "not cited"
                st.markdown(f"[{v.rank}] {mark} · C{v.chunk_id} · [report {v.report_id}]({v.source_url}) · {pb.safe_markdown(v.title)} · *{v.section}*")
            st.caption(f"Model: {ans.model}")

        st.markdown(f"### Retrieved chunks - {run.mode}, chunk set {run.chunk_set}")
        st.caption("📨 = the top K chunks, the ONLY text the answer model is shown.  ✅ = given to the model AND cited in its answer.  "
                   "🔎 = ranked K+1 and below: found by retrieval, never shown to the model. Chunks that were near-duplicates of a better "
                   f"one, or beyond {cfg.retrieval.max_chunks_per_report} per report, are removed before ranking and are not listed.")
        if run.supplied:
            st.markdown(f"#### Given to the answer model (top {run.k})")
            for v in run.supplied:
                render_hit(v, run, cited)
        if run.extra:
            st.markdown(f"#### Retrieved but NOT given to the model (ranks {run.k + 1}-{len(run.views)})")
            for v in run.extra:
                render_hit(v, run, cited)
        if not run.views:
            st.warning("Nothing was retrieved. In bm25 mode this means no keyword of the question occurs in any chunk.")

        # ---- save ----
        st.markdown("### Evaluate and save")
        if ans:
            a1, a2 = st.columns(2)
            accuracy = a1.selectbox("Is the answer correct?", pb.ACCURACY, key=f"acc_{run.id}")
            faith = a2.selectbox("Is every claim supported by the sources shown?", pb.FAITHFULNESS, key=f"faith_{run.id}")
            if st.button("Run the LLM judge on faithfulness (costs one call)", disabled=judge_llm not in usable):
                with st.spinner("Judging..."):
                    try:
                        st.session_state["judge"] = be.judge(run, ans, judge_llm) or {"error": "the judge's reply could not be parsed"}
                    except Exception as exc:  # noqa: BLE001
                        st.session_state["judge"] = {"error": f"{type(exc).__name__}: {exc}"}
            if st.session_state.get("judge"):
                j = st.session_state["judge"]
                st.write("Judge:", "⚠️ " + j["error"] if "error" in j else
                         ("✅ faithful" if j["faithful"] else "❌ not faithful") + (f" - unsupported: {j['unsupported']}" if j.get("unsupported") else ""))
        else:
            accuracy, faith = pb.ACCURACY[0], pb.FAITHFULNESS[0]
            st.caption("Press 'Retrieve + answer' to also rate the answer; you can still save the chunk marks below.")
        notes = st.text_input("Notes", key=f"notes_{run.id}")
        labels = {v.chunk_id: st.session_state.get(f"rel_{run.id}_{v.chunk_id}", "unjudged") for v in run.views}
        judged = sum(1 for x in labels.values() if x != "unjudged")
        st.caption(f"{judged} of {len(labels)} chunks marked.")
        if judged:
            live = pb.record_metrics(pb.build_record(run, labels, ans, accuracy, faith, notes))
            cols = st.columns(5)
            for col, name in zip(cols, [f"P@{min(run.k, 5)}", f"R@{min(run.k, 5)}", "MRR", f"nDCG@{min(run.k, 5)}", f"judged@{min(run.k, 5)}"]):
                col.metric(name, fmt(live.get(name), 3))
            st.caption("Unjudged chunks count as not relevant. Recall is relative to the relevant chunks you marked in this list.")
        if st.button("💾 Save evaluation", disabled=judged == 0 and not ans):
            rec = pb.build_record(run, labels, ans, accuracy, faith, notes, st.session_state.get("judge"), st.session_state.get("ans_profile", ""))
            log.append(rec)
            st.session_state["saved"] = str(log.path)
        if st.session_state.get("saved"):
            st.success(f"Saved to {st.session_state['saved']}. See the Evaluation tab.")


# ================= Chunk Explorer =================
with tab_chunks:
    st.subheader(f"Reports and how they were chunked - chunk set {chunk_set}")
    cs_cfg = be.chunk_set_config(chunk_set) or {}
    min_tokens = int(cs_cfg.get("min_tokens", 0))
    chunks = be.chunks(chunk_set)
    reports = be.reports()
    by_report: dict[int, list] = {}
    for c in chunks:
        by_report.setdefault(c.report_id, []).append(c)
    st.caption(f"Chunk settings: {cs_cfg}. Token counts are the project's ESTIMATE (characters / 3.6), not a tokenizer's.")

    table = df([{"report_id": r.report_id, "title": r.title, "program": r.program, "weakness": r.weakness, "source": r.source or "dataset",
                 "chars": len(r.body), "chunks": len(by_report.get(r.report_id, [])),
                 "tokens (est.)": sum(c.n_tokens for c in by_report.get(r.report_id, []))} for r in reports.values()])
    f1, f2, f3 = st.columns(3)
    text_filter = f1.text_input("Filter title", "")
    prog = f2.selectbox("Program", ["(all)"] + sorted(table["program"].unique()))
    weak = f3.selectbox("Weakness", ["(all)"] + sorted(w for w in table["weakness"].unique() if w))
    view = table
    if text_filter:
        view = view[view["title"].str.contains(text_filter, case=False, regex=False)]
    if prog != "(all)":
        view = view[view["program"] == prog]
    if weak != "(all)":
        view = view[view["weakness"] == weak]
    st.dataframe(view, hide_index=True)
    st.caption(f"{len(view)} of {len(table)} reports")

    if len(view):
        pick = st.selectbox("Inspect report", view["report_id"].tolist(), format_func=lambda i: f"{i} - {reports[i].title[:90]}")
        rep = reports[pick]
        rc = by_report.get(pick, [])
        st.markdown(f"**{pb.safe_markdown(rep.title)}**  \n[{rep.source_url or 'https://hackerone.com/reports/' + str(pick)}]({rep.source_url or 'https://hackerone.com/reports/' + str(pick)})  \n"
                    f"`{rep.program}` · {rep.weakness or 'no weakness'} · severity {rep.severity or '-'} · {rep.substate} · quality flags {rep.flags} · {len(rep.body)} characters")
        if rep.when_to_try:
            st.caption("'When to try this' (embedded with every chunk): " + rep.when_to_try)
        with st.expander("Full report text (as stored after cleaning)"):
            st.text(rep.body)
        if not rc:
            st.warning("This report has no chunks in this chunk set.")
        for i, c in enumerate(rc):
            warn = pb.chunk_checks(c, min_tokens)
            with st.expander(f"chunk {c.ordinal + 1}/{len(rc)} · C{c.chunk_id} · {c.section} · {c.n_tokens} est. tokens" + (" · ⚠️ " + "; ".join(warn) if warn else ""),
                             expanded=len(rc) <= 3):
                st.caption("Header embedded with this chunk:")
                st.text(c.header)
                st.caption("Chunk text:")
                st.text(c.text)
                if i + 1 < len(rc):
                    end, start = pb.overlap_preview(c, rc[i + 1])
                    st.caption("Boundary with the next chunk - end of this one, then start of the next (overlap shows the same words twice):")
                    st.text(end + "\n  ⟂⟂⟂ boundary ⟂⟂⟂\n" + start)

    st.divider()
    st.markdown("#### Chunking quality, all chunk sets")
    st.caption("Heuristics to look at, not verdicts: split code fences, chunks that start mid-sentence or end without sentence punctuation, very small chunks.")
    rows = []
    for name in chunk_sets:
        sc = be.chunk_set_config(name) or {}
        s = pb.chunk_set_summary(be.chunks(name), int(sc.get("min_tokens", 0)))
        rows.append({"chunk_set": name, **s})
    st.dataframe(df(rows), hide_index=True)


# ================= Evaluation =================
with tab_eval:
    st.subheader("Evaluation dashboard")
    records = log.load()
    if not records:
        st.info("Nothing saved yet. In the Ask tab: ask, mark chunks relevant / irrelevant, rate the answer, press 'Save evaluation'.")
    else:
        ks = [x for x in pb.KS if x <= max(r["pool"] for r in records)]
        st.markdown(f"**{len(records)} saved evaluations** from {len({r['question'] for r in records})} distinct questions.")
        st.caption("Precision@K = relevant chunks in the top K ÷ K.  Recall@K = relevant in the top K ÷ all chunks you marked relevant for that question.  "
                   "MRR = 1 ÷ rank of the first relevant chunk.  nDCG@K rewards putting relevant chunks higher.  All computed with the project's "
                   "own metric functions on YOUR marks; unjudged chunks count as not relevant, so check judged@K. With few questions, treat "
                   "differences as noise.")
        st.markdown("#### Retrieval quality by configuration")
        agg = df(pb.aggregate(records, ks))
        st.dataframe(agg, hide_index=True)

        st.markdown("#### Answer quality (your ratings)")
        s = pb.answer_summary(records)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Answers rated", s["rated_accuracy"])
        c2.metric("Rated correct", fmt(s["accuracy_correct_rate"], 2))
        c3.metric("Faithfulness rated", s["rated_faithfulness"])
        c4.metric("Fully supported", fmt(s["faithful_rate"], 2))
        st.write({"accuracy": s["accuracy"], "faithfulness": s["faithfulness"]})

        st.markdown("#### Every saved evaluation")
        flat = []
        for r in records:
            m = pb.record_metrics(r, ks)
            a = r.get("answer") or {}
            flat.append({"saved_at": r["saved_at"], "question": r["question"], "configuration": pb.config_key(r), "k": r["k"],
                         **{n: m.get(n) for n in [f"P@{x}" for x in ks] + [f"R@{x}" for x in ks] + ["MRR"] + [f"nDCG@{x}" for x in ks]},
                         "latency_ms": m["latency_ms"], "answer_accuracy": a.get("accuracy"), "faithfulness": a.get("faithfulness"),
                         "judge_faithful": (a.get("judge") or {}).get("faithful"), "notes": r.get("notes", "")})
        flat_df = df(flat)
        st.dataframe(flat_df, hide_index=True)
        st.download_button("Download as CSV", flat_df.to_csv(index=False).encode("utf-8"), "playground_evaluations.csv", "text/csv")

        st.markdown("#### Compare chunk sets and retrieval modes on your saved questions")
        st.caption("Re-runs each saved question under other settings and scores it against the REPORTS you marked relevant (report level, because "
                   "chunk ids differ between chunk sets). Costs at most one embedding call per distinct question. No answer model is called.")
        usable_sets = [n for n in chunk_sets if be.embedded_count(n, embed_profile) == info["chunks_per_set"][n]]
        cc1, cc2, cc3 = st.columns(3)
        pick_sets = cc1.multiselect("Chunk sets", chunk_sets, default=[chunk_set])
        pick_modes = cc2.multiselect("Modes", list(pb.MODES), default=["bm25"] + (["hybrid"] if vectors_ready else []))
        cmp_k = cc3.number_input("K for the comparison", 1, 20, int(k), key="cmp_k")
        needs_vec = any(m in ("vector", "hybrid") for m in pick_modes)
        missing = [n for n in pick_sets if needs_vec and n not in usable_sets]
        if missing:
            st.warning(f"Not embedded for '{embed_profile}': {missing}. Run  hrag.bat embed --set <name>  or choose bm25 only.")
        if st.button("Run comparison", disabled=not pick_sets or not pick_modes or bool(missing)):
            try:
                with st.spinner("Re-running your saved questions..."):
                    st.session_state["cmp"] = pb.compare_configs(be, records, chunk_sets=pick_sets, modes=pick_modes, k=int(cmp_k), embed_profile=embed_profile)
            except Exception as exc:  # noqa: BLE001
                st.error(f"{type(exc).__name__}: {exc}")
        if st.session_state.get("cmp"):
            st.dataframe(df(st.session_state["cmp"]), hide_index=True)


# ================= Report Library =================
def lib_chunk_card(v: pb.ChunkView, n: int, cited: set[int], mode: str) -> None:
    """One retrieved chunk, inside the selected report. Everything from the database is shown with st.text (never rendered)."""
    badge = "✅ cited in the answer" if v.chunk_id in cited else "📨 given to the model, not cited"
    with st.container(border=True):
        st.markdown(f"**[{n}] · {badge}**  \nC{v.chunk_id} · *{v.section}* · chunk {v.ordinal + 1} of this report · {v.n_tokens} est. tokens")
        st.text(f"Report {v.report_id} · {v.title}\nProgram: {v.program or '-'} · Weakness: {v.weakness or '-'}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Final score" + (" (RRF)" if mode == "hybrid" else ""), fmt(v.score, 5))
        c2.metric("Vector rank / cosine", f"{fmt(v.vector_rank)} / {fmt(v.vector_score, 3)}")
        c3.metric("BM25 rank / score", f"{fmt(v.bm25_rank)} / {fmt(v.bm25_score, 3)}")
        if mode == "hybrid":
            st.caption(f"RRF: 1/(60+vector rank) + 1/(60+BM25 rank) = {fmt(v.rrf_vector, 5)} + {fmt(v.rrf_bm25, 5)} (ranks are within this report)")
        st.caption("Header embedded with this chunk:")
        st.text(v.header)
        st.caption("Complete chunk text (exactly as stored; never rendered as markdown or HTML):")
        st.text(v.text)


with tab_lib:
    st.subheader("Report Library")
    st.caption("Read the original report on HackerOne, form your own question, and see whether the RAG system finds and uses the right text of THAT report.")
    lib_reports = be.reports()
    lib_by_report: dict[int, list] = {}
    for c in be.chunks(chunk_set):
        lib_by_report.setdefault(c.report_id, []).append(c)
    try:
        with st.spinner("Reading the dataset's original texts (first time only)..."):
            lib_originals = be.originals()
    except Exception:  # noqa: BLE001 - the comparison is optional; the library works without it
        lib_originals = {}

    lib_query = st.text_input("Search by report ID, title, program or weakness", key="lib_query",
                              placeholder="e.g. 534450 · ssrf · shopify · cross-site scripting (every word must match somewhere)")
    found = pb.search_reports(lib_reports.values(), lib_query)
    st.caption(f"{len(found)} of {len(lib_reports)} reports")
    if found:
        st.dataframe(df([{"report_id": r.report_id, "title": r.title, "program": r.program, "weakness": r.weakness,
                          "disclosed": r.disclosed_at[:10], "chunks": len(lib_by_report.get(r.report_id, []))} for r in found]),
                     hide_index=True, height=240)
    st.download_button("⬇ Download all reports (with HackerOne URLs) as CSV", pb.reports_csv(lib_reports.values(), lib_originals),
                       file_name="hackerone_reports.csv", mime="text/csv")

    if not found:
        st.info("No report matches. Clear the search box to list all reports.")
    else:
        pick = st.selectbox("Open report", [r.report_id for r in found], key="lib_pick",
                            format_func=lambda i: f"{i} · {lib_reports[i].title[:90]}")
        rep = lib_reports[pick]
        url = pb.report_url(rep.report_id)                    # built from the integer id only
        rc = lib_by_report.get(pick, [])
        head, link = st.columns([5, 2])
        head.subheader(pb.safe_markdown(rep.title))
        link.link_button("🔗 Open original on HackerOne", url, type="primary")
        st.text(f"Report {rep.report_id} · {rep.program} · {rep.weakness or 'no weakness recorded'} · disclosed {rep.disclosed_at[:10]} · "
                f"{rep.vote_count} votes · {rep.substate}\n{url}\nquality flags: {rep.flags or 'none'}")

        original = lib_originals.get(pick)
        if original is None:
            st.caption("The dataset files were not found in the cache folder, so the stored text cannot be compared with the dataset's original.")
        else:
            diff = pb.cleaning_diff(original, rep.body)
            if diff["identical"]:
                st.success("The stored text is identical to the dataset's original (ignoring whitespace).")
            else:
                st.warning(f"The stored text differs from the dataset's original: {diff['kept_share']:.0%} of its characters are kept. Cleaning masks cookies "
                           "and tokens and removes hidden characters and AI-directed instructions, so a question whose answer sat in the removed "
                           "text cannot be answered by the RAG system.")
                with st.expander("Show exactly what changed (- original, + stored; invisible characters shown as <U+....>)"):
                    st.text(diff["diff"])
        st.caption("The dataset is a snapshot: the live HackerOne page may differ from both, and screenshots or attachments are not in the dataset.")

        st.markdown(f"##### Complete report text stored in SQLite ({len(rep.body):,} characters)")
        with st.container(height=480, border=True):
            st.text(rep.body)

        # ---- one-click RAG restricted to this report ----
        st.divider()
        st.markdown("#### Test a question on this report")
        lib_mode = "hybrid" if vectors_ready else "bm25"
        st.caption(f"One click runs the project's real {lib_mode} search over only this report's {len(rc)} chunks (chunk set {chunk_set}), gives the top "
                   f"{int(k)} to {cfg.llms[answer_llm].model} and shows its answer. Scores here are computed inside this report, so BM25 scores are "
                   "not comparable with the Ask tab's.")
        if not vectors_ready:
            st.warning("Hybrid search is not available because the chunks are not embedded yet, so this panel uses BM25 only (keyword search).")
        if 0 < len(rc) <= int(k):
            st.info(f"This report has {len(rc)} chunk(s) and K = {int(k)}, so ALL of them go to the model. This then tests answer generation more than "
                    "ranking. Lower K in the sidebar to make the ranking matter.")
        lib_question = st.text_area("Your question about this report", key=f"lib_q_{pick}", height=80,
                                    placeholder="e.g. Which parameter was vulnerable and what request proves it?")
        if st.button("Generate Answer", type="primary", key=f"lib_go_{pick}", disabled=not (can_answer and rc),
                     help=None if can_answer else "The selected answer model has no API key (see the sidebar)"):
            if not lib_question.strip():
                st.error("Type a question first.")
            else:
                st.session_state["lib_result"] = None
                try:
                    with st.spinner("Searching inside this report..."):
                        lrun = be.retrieve(lib_question, chunk_set=chunk_set, mode=lib_mode, k=int(k), pool=int(k), embed_profile=embed_profile,
                                           report_id=pick)
                    st.session_state["lib_result"] = {"pick": pick, "run": lrun, "ans": None, "error": None}
                    with st.spinner(f"Asking {cfg.llms[answer_llm].model}..."):
                        st.session_state["lib_result"]["ans"] = be.answer(lrun, answer_llm)
                except Exception as exc:  # noqa: BLE001
                    if st.session_state.get("lib_result"):
                        st.session_state["lib_result"]["error"] = f"{type(exc).__name__}: {exc}"
                    else:
                        st.error(f"{type(exc).__name__}: {exc}")

        lres = st.session_state.get("lib_result")
        if lres and lres["pick"] == pick:
            lrun, lans = lres["run"], lres["ans"]
            lcited = set(lans.cited_valid) if lans else set()
            st.markdown("### Answer")
            st.text("Question: " + lrun.question)
            if lres["error"]:
                st.error("The answer model failed, but the retrieved chunks are shown below. " + lres["error"])
            if lans:
                if lans.result.abstained:
                    st.warning("The model abstained: it says this report's chunks do not contain the answer.")
                lnotice = pb.answer_notice(lans.result)
                if lnotice:
                    (st.warning if lnotice[0] == "warning" else st.info)(lnotice[1])
                if lans.result.parse_error:
                    st.code(lans.display_text, language=None)        # the raw reply, exactly as received (never rendered as markdown)
                else:
                    st.markdown(pb.safe_markdown(lans.display_text))
                if lans.cited_invalid:
                    st.error("Cited chunks that were NOT given to the model (treat as invented): " + ", ".join(f"C{c}" for c in lans.cited_invalid))
                if lans.result.citations and not lans.inline_ids:
                    st.caption("The model listed citations but wrote no [..] markers in the answer text.")
                st.markdown("**Citations** (numbers match the [n] in the answer)")
                for n, v in enumerate(lrun.supplied, start=1):
                    st.text(f"[{n}] {'cited    ' if v.chunk_id in lcited else 'not cited'} · C{v.chunk_id} · {v.section} · chunk {v.ordinal + 1} of this report")
                st.caption(f"Model: {lans.model} · answer took {lans.llm_ms:.0f} ms")
            t = lrun.timings
            st.caption(f"Retrieval {t['total_ms']:.0f} ms (query embedding {'cached' if t['embed_cached'] else str(round(t['embed_ms'])) + ' ms'}; "
                       f"search {t['search_ms']:.0f} ms) · mode {lrun.mode} · K = {lrun.k}")
            st.markdown(f"### Retrieved chunks - restricted to report {lrun.scope_report_id}")
            st.caption("📨 = given to the model (the only text it saw). ✅ = also cited in its answer.")
            if not lrun.supplied:
                st.warning("Nothing was retrieved. In bm25 mode this means no keyword of the question occurs in this report's chunks.")
            for n, v in enumerate(lrun.supplied, start=1):
                lib_chunk_card(v, n, lcited, lrun.mode)

        # ---- the chunks of this report ----
        st.divider()
        st.markdown(f"#### Chunks of this report ({len(rc)}, chunk set {chunk_set})")
        if not rc:
            st.warning("This report has no chunks in this chunk set.")
        for i, c in enumerate(rc):
            warn = pb.chunk_checks(c, int((be.chunk_set_config(chunk_set) or {}).get("min_tokens", 0)))
            with st.expander(f"chunk {c.ordinal + 1}/{len(rc)} · C{c.chunk_id} · {c.section} · {c.n_tokens} est. tokens" + (" · ⚠️ " + "; ".join(warn) if warn else "")):
                st.caption("Header embedded with this chunk:")
                st.text(c.header)
                st.caption("Chunk text:")
                st.text(c.text)
                if i + 1 < len(rc):
                    end, start = pb.overlap_preview(c, rc[i + 1])
                    st.caption("Boundary with the next chunk (overlap shows the same words twice):")
                    st.text(end + "\n  ⟂⟂⟂ boundary ⟂⟂⟂\n" + start)
