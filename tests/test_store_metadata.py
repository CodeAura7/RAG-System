"""Reports carry source/quality/enrichment fields; every chunk carries a metadata snapshot; old databases upgrade in place."""
import json
import sqlite3

from hrag.store import NewChunk, ReportRow, Store

OLD_SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE reports(
  report_id INTEGER PRIMARY KEY,
  title TEXT NOT NULL, program TEXT NOT NULL DEFAULT '', weakness TEXT NOT NULL DEFAULT '',
  substate TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL DEFAULT '', disclosed_at TEXT NOT NULL DEFAULT '',
  severity TEXT NOT NULL DEFAULT '', vote_count INTEGER NOT NULL DEFAULT 0,
  body TEXT NOT NULL, body_hash TEXT NOT NULL
);
CREATE TABLE chunk_sets(name TEXT PRIMARY KEY, config_json TEXT NOT NULL);
CREATE TABLE chunks(
  chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_set TEXT NOT NULL REFERENCES chunk_sets(name) ON DELETE CASCADE,
  report_id INTEGER NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL, section TEXT NOT NULL, has_code INTEGER NOT NULL, n_tokens INTEGER NOT NULL,
  header TEXT NOT NULL, text TEXT NOT NULL
);
CREATE TABLE embeddings(
  chunk_id INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  profile TEXT NOT NULL, model TEXT NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL,
  PRIMARY KEY(chunk_id, profile)
);
"""


def full_report(rid=1) -> ReportRow:
    return ReportRow(
        report_id=rid, title="SSRF in webhooks", program="gitlab", weakness="Server-Side Request Forgery", substate="resolved",
        created_at="2023-01-01", disclosed_at="2023-06-01", severity="high", vote_count=3, body="body text", body_hash="h",
        source="gitlab", source_url="https://gitlab.com/gitlab-org/gitlab/-/issues/1", cwe="CWE-918",
        quality_flags=json.dumps(["no_impact_section"]), quality_score=0.75, when_to_try="Try when an app fetches user-supplied URLs.",
        images=json.dumps([{"kind": "image", "ref": "shot.png"}]), injection_hits=2)


def test_new_report_fields_round_trip():
    store = Store(":memory:")
    store.replace_reports([full_report()])
    got = store.report(1)
    assert got == full_report()
    assert got.flags == ["no_impact_section"] and got.image_list == [{"kind": "image", "ref": "shot.png"}]


def test_old_style_report_rows_still_construct_with_defaults():
    r = ReportRow(report_id=2, title="t", program="", weakness="", substate="", created_at="", disclosed_at="", severity="",
                  vote_count=0, body="b", body_hash="h")
    assert r.source == "" and r.flags == [] and r.quality_score == 0.0 and r.when_to_try == "" and r.injection_hits == 0


def test_every_chunk_stores_and_returns_its_metadata():
    store = Store(":memory:")
    store.replace_reports([full_report()])
    meta = {"program": "gitlab", "weakness": "SSRF", "source": "gitlab", "quality_flags": ["no_impact_section"], "section": "steps"}
    store.replace_chunk_set("c400", {"max_tokens": 400}, [
        NewChunk(1, 0, "steps", True, 50, "HEADER", "text a", meta), NewChunk(1, 1, "impact", False, 20, "HEADER", "text b")])
    a, b = store.chunks("c400")
    assert a.meta == meta
    assert b.meta == {}                                                     # chunks built without metadata still load


def test_when_to_try_can_be_updated_without_touching_the_rest():
    store = Store(":memory:")
    store.replace_reports([full_report()])
    store.set_when_to_try(1, "New description")
    got = store.report(1)
    assert got.when_to_try == "New description" and got.body == "body text" and got.flags == ["no_impact_section"]


def test_an_old_database_is_upgraded_in_place_and_keeps_its_data(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript(OLD_SCHEMA)
    con.execute("INSERT INTO reports(report_id,title,body,body_hash) VALUES(5,'Old report','old body','x')")
    con.execute("INSERT INTO chunk_sets VALUES('c400','{}')")
    con.execute("INSERT INTO chunks(chunk_set,report_id,ordinal,section,has_code,n_tokens,header,text) VALUES('c400',5,0,'body',0,5,'H','t')")
    con.commit()
    con.close()

    store = Store(path)                                                      # opening upgrades the schema
    old = store.report(5)
    assert old.title == "Old report" and old.source == "" and old.flags == []
    assert store.chunks("c400")[0].meta == {}
    store.replace_reports([full_report(6)])                                  # and the new columns are writable
    assert store.report(6).source == "gitlab"
    Store(path)                                                              # opening twice must not fail (idempotent)
