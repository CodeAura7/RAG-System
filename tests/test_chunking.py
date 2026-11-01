import re

from hrag.chunking import chunk_report, classify_section, estimate_tokens, make_header, parse_units

SHORT = "Title line\n\n## Steps To Reproduce\n1. Open /login\n2. Send the payload\n\n## Impact\nAccount takeover."


def balanced_fences(text: str) -> bool:
    return len(re.findall(r"^```", text, flags=re.M)) % 2 == 0


def test_short_report_stays_one_chunk_with_original_text():
    chunks = chunk_report(SHORT, max_tokens=400, overlap_tokens=50)
    assert len(chunks) == 1 and chunks[0].text == SHORT and chunks[0].ordinal == 0


def test_empty_body_gives_no_chunks():
    assert chunk_report("   \n\n ", max_tokens=400) == []


def test_section_classification():
    assert classify_section("Steps To Reproduce:") == "steps"
    assert classify_section("Proof of Concept") == "steps"
    assert classify_section("Impact") == "impact"
    assert classify_section("Possible Solution") == "fix"
    assert classify_section("Vulnerability Details") == "summary"
    assert classify_section("Supporting material") == "other"


def test_heading_styles_are_detected():
    body = "Intro text\n\nSteps to reproduce:\nstep\n\n**Impact**\nbad\n\nFix\n===\nupgrade\n\n# Poc\nrun it"
    labels = [u.text for u in parse_units(body) if u.kind == "heading"]
    assert labels == ["## Steps to reproduce", "## Impact", "## Fix", "## Poc"]


def test_code_block_that_fits_is_never_split():
    code = "```\n" + "\n".join(f"GET /a{i} HTTP/1.1" for i in range(15)) + "\n```"
    body = "Summary text. " * 60 + "\n\n## Steps To Reproduce\n1. Send this:\n\n" + code + "\n\n## Impact\nBad. " * 3
    chunks = chunk_report(body, max_tokens=300, overlap_tokens=30)
    holders = [c for c in chunks if "GET /a0 HTTP/1.1" in c.text]
    assert len(holders) == 1 and "GET /a14 HTTP/1.1" in holders[0].text and holders[0].has_code
    assert all(balanced_fences(c.text) for c in chunks)


def test_oversized_code_block_is_split_into_valid_fenced_pieces():
    code = "```python\n" + "\n".join(f"print({i}) # " + "x" * 40 for i in range(120)) + "\n```"
    chunks = chunk_report("PoC:\n\n" + code, max_tokens=200)
    assert len(chunks) > 1
    assert all(balanced_fences(c.text) for c in chunks)
    assert all(c.n_tokens <= 200 * 1.2 for c in chunks)
    joined = "\n".join(c.text for c in chunks)
    assert "print(0)" in joined and "print(119)" in joined


def test_unclosed_fence_does_not_swallow_or_crash():
    chunks = chunk_report("Intro\n\n```\ncurl http://x\nstill code", max_tokens=400)
    assert len(chunks) == 1 and chunks[0].has_code


def test_long_prose_splits_with_overlap_and_respects_budget():
    sentences = [f"Sentence number {i} explains the bug in some detail here." for i in range(80)]
    chunks = chunk_report(" ".join(sentences), max_tokens=120, overlap_tokens=30)
    assert len(chunks) > 3
    assert all(c.n_tokens <= 120 + 5 for c in chunks)
    for left, right in zip(chunks, chunks[1:]):                 # every break is overlapped
        last_sentence = left.text.strip().splitlines()[-1]
        assert last_sentence in right.text
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_no_chunk_ends_on_a_bare_heading_and_sections_break_cleanly():
    body = ("Intro. " * 40 + "\n\n## Steps To Reproduce\n" + "Do this thing. " * 40 + "\n\n## Impact\n" + "Very bad. " * 40)
    chunks = chunk_report(body, max_tokens=150, overlap_tokens=0, min_tokens=20)
    assert len(chunks) >= 3
    assert not any(c.text.rstrip().splitlines()[-1].startswith("## ") for c in chunks)
    assert any(c.section == "impact" for c in chunks) and any(c.section == "steps" for c in chunks)


def test_whole_report_set_never_splits():
    big = "word " * 5000
    assert len(chunk_report(big, max_tokens=100000)) == 1


def test_header_and_estimator():
    h = make_header(42, "XSS in search", "acme", "Cross-site Scripting (XSS)", "resolved", "steps")
    assert h.startswith("HackerOne report #42: XSS in search") and "section: steps" in h
    assert estimate_tokens("") == 0 and estimate_tokens("abcd" * 9) > 9


# ---------- requests and payloads stay with the step that introduces them ----------
def _steps_report(intro: str, code: str, filler_sentences: int = 14) -> str:
    filler = " ".join(f"Background sentence number {i} explains the setup." for i in range(filler_sentences))
    return f"## Steps To Reproduce\n\n{filler}\n\n{intro}\n\n{code}\n\n3. Observe the response."


REQUEST = "```http\nPOST /api/v4/projects/5/releases HTTP/1.1\nHost: gitlab.example.vm\nContent-Type: application/json\n\n{\"tag_name\":\"x\"}\n```"


def test_a_step_that_introduces_a_request_is_never_separated_from_it():
    for intro in ("2. Send this request:", "2. Then run the following command", "Payload used in the second step:"):
        chunks = chunk_report(_steps_report(intro, REQUEST), max_tokens=130, overlap_tokens=0, min_tokens=0)
        assert len(chunks) >= 2
        holder = next(c for c in chunks if "POST /api/v4/projects/5/releases" in c.text)
        # a short "Label:" line is parsed as a section heading (colon dropped); either way it stays with its request
        assert intro.rstrip(":") in holder.text and holder.has_code, f"'{intro}' was split from its request"


def test_the_glue_is_skipped_when_step_plus_request_cannot_fit_one_chunk():
    big = "```\n" + "\n".join(f"line {i} of a long captured response body" for i in range(60)) + "\n```"
    chunks = chunk_report(_steps_report("2. Send this request:", big), max_tokens=100, overlap_tokens=0, min_tokens=0)
    assert all(estimate_tokens(c.text) <= 120 for c in chunks)               # nothing blows the budget
    assert balanced_fences("\n".join(c.text for c in chunks))


def test_header_carries_severity_and_when_to_try_without_breaking_the_old_format():
    old = make_header(42, "XSS in search", "acme", "Cross-site Scripting (XSS)", "resolved", "steps")
    assert old == "HackerOne report #42: XSS in search | program: acme | weakness: Cross-site Scripting (XSS) | outcome: resolved | section: steps"
    new = make_header(42, "XSS in search", "acme", "XSS", "resolved", "steps", severity="high", when_to_try="Try it on search boxes.")
    first, second = new.split("\n")
    assert "severity: high" in first and first.endswith("section: steps") and second == "When to try this: Try it on search boxes."
