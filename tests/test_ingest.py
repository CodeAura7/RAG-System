import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hrag.config import IngestSettings
from hrag.ingest import (IngestError, download_dataset, jaccard, read_rows, select_reports, shingles,
                         to_report_row, write_summary)

SETTINGS = IngestSettings(n_reports=5, seed=7, min_body_chars=100, allowed_substates=("resolved",),
                          near_duplicate_jaccard=0.8, dataset_base_url="https://data.example/files/",
                          dataset_files=("a.parquet",))


def body(topic: str, extra: str = "") -> str:
    sentences = [f"Step {i} of the {topic} attack abuses parameter {i} in an unusual way, number {i * 7}, which the server accepts blindly."
                 for i in range(8)]
    return " ".join(sentences) + extra


def row(rid, text, substate="resolved", original=None, title=None):
    return {"id": rid, "title": title or f"Report {rid}", "created_at": "2023-01-01", "substate": substate,
            "vulnerability_information": text, "team": {"handle": "acme"}, "disclosed_at": "2023-02-01",
            "weakness": {"id": 1, "name": "XSS"}, "original_report_id": original, "vote_count": 3,
            "structured_scope": {"asset_identifier": "x", "max_severity": "high"}}


def messy_rows():
    rows = [
        row(1, ""), row(2, "tiny"), row(3, body("a"), substate="not-applicable"),
        row(4, body("b"), original=999), row(5, body("c"), substate="duplicate"),
        row(6, body("d")), row(7, body("d")),                                   # exact duplicate of 6
        row(8, body("e")), row(9, body("e", " one extra sentence added at the end.")),   # near duplicate of 8
    ]
    rows += [row(100 + i, body(f"topic{i}", f" unique tail {i} " * 30)) for i in range(12)]
    return rows


def test_filters_and_dedupe_are_counted_and_nothing_bad_is_selected():
    selected, summary = select_reports(messy_rows(), SETTINGS, n=50)
    ids = {r.report_id for r in selected}
    assert not ids & {1, 2, 3, 4, 5}
    assert (6 in ids) != (7 in ids)                          # exactly one of the exact twins
    assert (8 in ids) != (9 in ids)                          # exactly one of the near twins
    rej = summary["rejected"]
    assert rej["empty_body"] == 1 and rej["body_too_short"] == 1 and rej["substate_not_allowed"] == 1
    assert rej["marked_duplicate_by_hackerone"] == 2 and rej["exact_duplicate"] == 1 and rej["near_duplicate"] == 1
    assert summary["selected"] == len(selected) == len(ids) and selected == sorted(selected, key=lambda r: r.report_id)


def test_selection_is_reproducible_and_seed_sensitive():
    a, _ = select_reports(messy_rows(), SETTINGS, n=5, seed=1)
    b, _ = select_reports(messy_rows(), SETTINGS, n=5, seed=1)
    c, _ = select_reports(messy_rows(), SETTINGS, n=5, seed=2)
    assert [r.report_id for r in a] == [r.report_id for r in b]
    assert [r.report_id for r in a] != [r.report_id for r in c]
    assert len(a) == 5


def test_asking_for_more_than_exists_returns_what_exists():
    selected, summary = select_reports(messy_rows(), SETTINGS, n=1000)
    assert 0 < len(selected) < 1000 and summary["requested"] == 1000


def test_metadata_extraction_handles_dicts_lists_and_missing_values():
    r = to_report_row({**row(1, body("x")), "structured_scope": [{"max_severity": None}, {"max_severity": "critical"}],
                       "team": None, "weakness": {"name": "SQL Injection"}, "vote_count": None})
    assert r.severity == "critical" and r.program == "" and r.weakness == "SQL Injection" and r.vote_count == 0


def test_shingles_and_jaccard_behave():
    a, b = shingles("one two three four five six seven"), shingles("one two three four five six eight")
    assert 0 < jaccard(a, b) < 1 and jaccard(a, a) == 1.0 and jaccard(set(), a) == 0.0


def test_parquet_roundtrip_with_nested_columns(tmp_path):
    table = pa.Table.from_pylist(messy_rows()[:10] + [row(500, body("zzz"))])
    path = tmp_path / "a.parquet"
    pq.write_table(table, path)
    rows = read_rows([path])
    assert len(rows) == 11 and rows[0]["team"]["handle"] == "acme"
    selected, _ = select_reports(rows, SETTINGS, n=50)
    assert selected and all(r.program == "acme" and r.severity == "high" for r in selected)


def test_download_writes_validates_and_skips_existing(tmp_path):
    table = pa.Table.from_pylist([row(1, body("a"))])
    src = tmp_path / "src.parquet"
    pq.write_table(table, src)
    payload = src.read_bytes()
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, content=payload)

    cache = tmp_path / "cache"
    client = httpx.Client(transport=httpx.MockTransport(handler))
    logs = []
    paths = download_dataset(SETTINGS, cache, client=client, log=logs.append)
    assert paths == [cache / "a.parquet"] and calls == ["https://data.example/files/a.parquet"]
    download_dataset(SETTINGS, cache, client=client, log=logs.append)
    assert len(calls) == 1                                    # second run reused the file


def test_download_rejects_non_parquet_and_http_errors(tmp_path):
    bad = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"<html>not parquet</html>")))
    with pytest.raises(IngestError, match="valid parquet"):
        download_dataset(SETTINGS, tmp_path / "c1", client=bad, log=lambda s: None)
    assert not list((tmp_path / "c1").glob("*.part"))
    err = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(IngestError, match="404"):
        download_dataset(SETTINGS, tmp_path / "c2", client=err, log=lambda s: None)


def test_summary_files(tmp_path):
    _, summary = select_reports(messy_rows(), SETTINGS, n=3)
    ids_path, json_path = write_summary(summary, tmp_path)
    assert ids_path.read_text().split() == [str(i) for i in summary["report_ids"]] and json_path.is_file()
