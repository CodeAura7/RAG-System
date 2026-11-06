"""tools/import_selection.py: keeps every id you chose, tops up from the most-voted rest, writes only its own files."""
from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hrag.config import load_config
from hrag.store import Store
from tools import import_selection as imp

PROJECT = Path(__file__).resolve().parents[1]


def body(rid: int, words: int = 150) -> str:
    """Long enough to survive cleaning, and different enough from every other report to avoid the duplicate filters."""
    filler = " ".join(f"w{rid}x{k}" for k in range(words))
    return (f"## Summary\nReport {rid} describes a flaw. {filler}\n\n## Steps to reproduce\n1. open the page {rid}\n2. send the request\n"
            f"\n```\nGET /r{rid} HTTP/1.1\n```\n\n## Impact\nAn attacker can read data of user {rid}.")


def raw(rid: int, *, program: str | None = None, votes: int = 0, substate: str = "resolved", text: str | None = None,
        original: int | None = None, weakness: str | None = None, title: str | None = None) -> dict:
    return {"id": rid, "title": title or f"Title {rid}", "created_at": "2020-01-01T00:00:00Z", "substate": substate,
            "vulnerability_information": body(rid) if text is None else text, "team": {"handle": program or f"prog{rid}"},
            "disclosed_at": "2020-02-01T00:00:00Z", "weakness": {"name": weakness or f"weak{rid}"}, "original_report_id": original,
            "vote_count": votes, "structured_scope": [{"max_severity": "high"}]}


@pytest.fixture()
def cfg():
    return load_config(PROJECT / "config.toml")


# ---------------------------------------------------------------------------------------------- reading ids
def test_read_ids_text_and_csv(tmp_path):
    t = tmp_path / "ids.txt"
    t.write_text("# comment\n11\n22, 33  # trailing\n\n11\n", encoding="utf-8")
    assert imp.read_ids(t) == [11, 22, 33]                      # comments ignored, duplicate dropped, order kept
    c = tmp_path / "f.csv"
    c.write_bytes("﻿rank,id,title\n1,534450,Café â–ˆ\n2,1086108,x\n".encode("utf-8"))   # BOM + odd characters
    assert imp.read_ids(c) == [534450, 1086108]


def test_read_ids_rejects_bad_input(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("12\nabc\n", encoding="utf-8")
    with pytest.raises(ValueError, match="abc"):
        imp.read_ids(bad)
    empty = tmp_path / "empty.txt"
    empty.write_text("# nothing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no report ids"):
        imp.read_ids(empty)
    nocol = tmp_path / "x.csv"
    nocol.write_text("rank,url\n1,u\n", encoding="utf-8")
    with pytest.raises(ValueError, match="column named 'id'"):
        imp.read_ids(nocol)


# ---------------------------------------------------------------------------------------------- selection
def test_pinned_ids_are_all_kept_beyond_the_per_program_cap_and_topup_respects_it(cfg):
    cap = cfg.collect.per_program
    pinned = [raw(i, program="same") for i in range(1, cap + 3)]                     # more than the cap, one program
    others = [raw(100 + i, program="same", votes=500 - i) for i in range(5)]       # same program: must NOT be topped up
    others += [raw(200 + i, program=f"p{i}", votes=100 - i) for i in range(6)]      # distinct programs
    kept, s = imp.select_with_pinned(pinned + others, cfg, [r["id"] for r in pinned], n=len(pinned) + 4, pool_top_votes=50)
    ids = {r.report_id for r in kept}
    assert {r["id"] for r in pinned} <= ids                                          # every id you asked for is in
    assert s["pinned_kept"] == len(pinned) and s["pinned_dropped"] == []
    top = {t["id"] for t in s["topup_selected"]}
    assert len(top) == 4 and not any(100 <= i < 105 for i in top)                    # the capped program got nothing more
    assert top == {200, 201, 202, 203}                                               # equal quality: lowest id first (collect ignores votes)
    assert {r.source for r in kept} == {"dataset"}
    assert all(r.source_url == f"https://hackerone.com/reports/{r.report_id}" for r in kept)


def test_topup_pool_is_limited_to_the_most_voted_and_vote_counts_are_kept(cfg):
    pinned = [raw(1, votes=7)]
    rest = [raw(10 + i, votes=i) for i in range(20)]                                 # votes 0..19, ids 10..29
    kept, s = imp.select_with_pinned(pinned + rest, cfg, [1], n=4, pool_top_votes=3)
    assert {t["id"] for t in s["topup_selected"]} == {29, 28, 27}                    # only the top-3 voted could be chosen
    assert {r.report_id: r.vote_count for r in kept} == {1: 7, 27: 17, 28: 18, 29: 19}   # collect itself leaves votes at 0


def test_topup_skips_duplicates_and_unallowed_substates_but_pinned_ones_are_kept(cfg):
    rows = [raw(1, substate="duplicate", votes=1), raw(2, votes=5), raw(3, substate="informative", votes=900),
            raw(4, original=2, votes=900), raw(5, votes=3)]
    kept, s = imp.select_with_pinned(rows, cfg, [1], n=3, pool_top_votes=10)
    assert {r.report_id for r in kept} == {1, 2, 5}                                  # 3 and 4 never entered the pool
    assert s["pinned_marked_duplicate_by_hackerone"] == [1]                          # reported, not silently kept


def test_an_id_missing_from_the_dataset_is_reported_not_fatal(cfg):
    kept, s = imp.select_with_pinned([raw(1), raw(2, votes=9)], cfg, [1, 999], n=2, pool_top_votes=10)
    assert s["pinned_not_in_dataset"] == [999] and s["pinned_requested"] == 2 and s["pinned_kept"] == 1
    assert {r.report_id for r in kept} == {1, 2}


def test_dropped_ids_say_why(cfg):
    rows = [raw(10), raw(11, text=body(10)), raw(12, text="too short"), raw(13)]
    kept, s = imp.select_with_pinned(rows, cfg, [10, 11, 12, 13], n=10, pool_top_votes=0)
    reasons = {d["id"]: d["reason"] for d in s["pinned_dropped"]}
    assert reasons[12] == "too_short_after_cleaning"
    assert reasons[11] == "exact_duplicate_of_10"
    assert 10 not in reasons and 13 not in reasons
    assert s["pinned_kept"] == 2 and {r.report_id for r in kept} == {10, 13}


def test_a_near_duplicate_names_the_report_it_resembles_and_how_closely(cfg):
    twin = body(20).replace("w20x3 ", "changed ").replace("w20x90 ", "altered ")          # two words differ: not byte-identical
    assert twin != body(20)
    kept, s = imp.select_with_pinned([raw(20), raw(21, text=twin)], cfg, [20, 21], n=10, pool_top_votes=0)
    (dropped,) = s["pinned_dropped"]
    assert dropped["id"] == 21 and dropped["reason"].startswith("near_duplicate_of_20 (similarity 0.")
    assert [r.report_id for r in kept] == [20]


def test_over_quota_is_named_when_n_is_smaller_than_the_list(cfg):
    kept, s = imp.select_with_pinned([raw(1), raw(2), raw(3)], cfg, [1, 2, 3], n=2, pool_top_votes=0)
    assert len(kept) == 2 and len(s["pinned_dropped"]) == 1
    assert s["pinned_dropped"][0]["reason"].startswith("over_quota")


# ---------------------------------------------------------------------------------------------- the derived config
def test_derive_config_changes_only_the_two_paths_and_still_loads(tmp_path):
    original = (PROJECT / "config.toml").read_text(encoding="utf-8")
    derived = imp.derive_config(original, "data/hrag.selection.db", "eval_selection")
    p = tmp_path / "config.selection.toml"
    p.write_text(derived, encoding="utf-8")
    a, b = load_config(PROJECT / "config.toml"), load_config(p)
    assert b.db_path == tmp_path / "data/hrag.selection.db" and b.eval_dir == tmp_path / "eval_selection"
    assert b.cache_dir == tmp_path / "data/cache"                                    # the dataset cache stays shared
    for name in ("defaults", "retrieval", "collect", "ingest", "chunk_sets", "embeddings", "llms", "rerankers"):
        assert getattr(a, name) == getattr(b, name), name                           # nothing else moved
    body_lines = [ln for ln in derived.splitlines() if not ln.startswith("# Generated") and not ln.startswith("# reports live")
                  and not ln.startswith("#   python -m hrag --config")]
    assert [ln for ln in body_lines if ln.startswith("db ") or ln.startswith("eval_dir")] == \
        ['db        = "data/hrag.selection.db"', 'eval_dir  = "eval_selection"']


def test_derive_config_leaves_same_named_keys_in_other_sections_alone():
    src = '[paths]\ndb = "old.db"\neval_dir = "old"\n\n[other]\ndb = "keep-me"\neval_dir = "keep-me-too"\n'
    out = imp.derive_config(src, "new.db", "newev")
    assert 'db        = "new.db"' in out and 'eval_dir  = "newev"' in out
    assert 'db = "keep-me"' in out and 'eval_dir = "keep-me-too"' in out               # untouched, same text as before
    assert "old.db" not in out


def test_derive_config_adds_missing_keys_to_paths():
    out = imp.derive_config('[paths]\ncache_dir = "c"\n\n[defaults]\nx = 1\n', "d.db", "ev")
    assert 'db        = "d.db"' in out and 'eval_dir  = "ev"' in out
    assert out.index('db        = "d.db"') < out.index("[defaults]")


# ---------------------------------------------------------------------------------------------- the whole command
@pytest.fixture()
def project(tmp_path):
    """A throw-away project root: a copy of config.toml, three tiny parquet files, and an ids file."""
    shutil.copy(PROJECT / "config.toml", tmp_path / "config.toml")
    cache = tmp_path / "data" / "cache"
    cache.mkdir(parents=True)
    rows = [raw(i, votes=i) for i in range(1, 31)]
    names = load_config(tmp_path / "config.toml").ingest.dataset_files
    for name, chunk in zip(names, (rows[:10], rows[10:20], rows[20:])):
        pq.write_table(pa.Table.from_pylist(chunk), cache / name)
    (tmp_path / "ids.txt").write_text("1\n2\n3\n", encoding="utf-8")
    return tmp_path


def run(project, *extra):
    return imp.main(["--config", str(project / "config.toml"), "--ids", str(project / "ids.txt"), "--parquet-dir",
                     str(project / "data" / "cache"), "--n", "6", *extra])


def test_end_to_end_writes_only_its_own_files(project, capsys):
    assert run(project) == 0
    out = capsys.readouterr().out
    db = project / "data" / "hrag.selection.db"
    store = Store(db)
    assert len(store.report_ids()) == 6 and {1, 2, 3} <= store.report_ids()
    assert not (project / "data" / "hrag.db").exists()                               # the config's own database is never created
    summary = json.loads((project / "data" / "hrag.selection.summary.json").read_text(encoding="utf-8"))
    assert summary["pinned_kept"] == 3 and len(summary["topup_selected"]) == 3
    cfg2 = load_config(project / "config.selection.toml")
    assert cfg2.db_path == db and cfg2.eval_dir == project / "eval_selection"
    assert "--config config.selection.toml build" in out and "NOT from your list" in out


def test_second_run_refuses_without_force_and_replaces_with_it(project, capsys):
    assert run(project) == 0
    db = project / "data" / "hrag.selection.db"
    store = Store(db)
    store.db.execute("UPDATE reports SET title='EDITED' WHERE report_id=1")
    store.db.commit()
    store.db.close()
    capsys.readouterr()
    assert run(project) == 1 and "--force" in capsys.readouterr().out
    assert Store(db).report(1).title == "EDITED"                                     # the refusal changed nothing
    assert run(project, "--force") == 0
    assert Store(db).report(1).title == "Title 1"


def test_dry_run_writes_nothing(project, capsys):
    assert run(project, "--dry-run") == 0
    assert "nothing written" in capsys.readouterr().out
    assert not (project / "data" / "hrag.selection.db").exists()
    assert not (project / "config.selection.toml").exists()


def test_refuses_to_target_the_configured_database(project, capsys):
    assert run(project, "--out-db", "data/hrag.db") == 1
    assert "refusing" in capsys.readouterr().out
    assert not (project / "data" / "hrag.db").exists()


def test_existing_derived_config_is_kept_unless_forced(project, capsys):
    cfgfile = project / "config.selection.toml"
    cfgfile.write_text((project / "config.toml").read_text(encoding="utf-8") + "\n# my edit\n", encoding="utf-8")
    assert run(project) == 0
    assert "# my edit" in cfgfile.read_text(encoding="utf-8")                          # not overwritten
    out = capsys.readouterr().out
    assert "already existed, kept" in out
    assert "points at a different database" in out                                   # the kept file names data/hrag.db: warned
    assert run(project, "--force") == 0                                              # --force replaces the db AND the config
    assert load_config(cfgfile).db_path == project / "data" / "hrag.selection.db"
    assert "# my edit" not in cfgfile.read_text(encoding="utf-8")


def test_a_legacy_windows_console_cannot_crash_the_listing(project, monkeypatch):
    names = load_config(project / "config.toml").ingest.dataset_files
    rows = [raw(i, votes=i) for i in range(1, 30)] + [raw(30, votes=30, title="XSS \u2192 \u2603 \u2588\u2588")]
    for name, chunk in zip(names, (rows[:10], rows[10:20], rows[20:])):
        pq.write_table(pa.Table.from_pylist(chunk), project / "data" / "cache" / name)
    import io, sys
    raw_out = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw_out, encoding="cp1252", errors="strict"))   # what PowerShell may give
    assert run(project, "--dry-run", "--pool-top-votes", "1") == 0                   # pool = only the unicode-titled report
    sys.stdout.flush()
    assert "XSS \u2192 \u2603 \u2588\u2588".encode("utf-8") in raw_out.getvalue()      # printed intact as UTF-8, not crashed or dropped


def test_bad_ids_file_is_a_clean_error(project, capsys):
    (project / "ids.txt").write_text("1\nnope\n", encoding="utf-8")
    assert run(project) == 2
    assert "nope" in capsys.readouterr().err
