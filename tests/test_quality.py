"""Quality flags (incomplete / disputed / sparse / image-dependent ...) and the 0..1 score used to prefer good reports."""
from hrag import quality

FULL = """##### Summary

The export endpoint returns files of other tenants because the id is never checked against the session owner.

##### Steps to reproduce

1. Log in as user A.
2. Request the export:

```http
GET /export?id=42 HTTP/1.1
Host: app.example
```

3. Observe user B's file in the response.

##### Impact

An attacker can read every tenant's private files, which exposes customer data and credentials.
""" + "Additional detail about the root cause and the affected code path. " * 6


def test_a_complete_report_has_no_negative_flags_and_a_high_score():
    a = quality.assess(FULL)
    assert a.has_steps and a.has_poc and a.has_impact
    assert not ({"sparse", "no_steps", "no_poc", "no_impact", "incomplete", "disputed", "image_dependent"} & set(a.flags))
    assert a.score >= 0.85


def test_a_sparse_report_is_flagged_and_scores_low():
    a = quality.assess("It is vulnerable to XSS on the search page.")
    assert {"sparse", "no_steps", "no_poc", "no_impact", "incomplete"} <= set(a.flags)
    assert a.score < 0.3 and a.score < quality.assess(FULL).score


def test_missing_parts_are_flagged_individually():
    no_impact = FULL.split("##### Impact")[0]
    assert "no_impact" in quality.assess(no_impact).flags and "no_steps" not in quality.assess(no_impact).flags
    no_poc = FULL.replace("```http\nGET /export?id=42 HTTP/1.1\nHost: app.example\n```", "")
    flags = quality.assess(no_poc).flags
    assert "no_poc" in flags and "incomplete" not in flags                    # steps still exist


def test_disputed_from_outcome_labels_and_comments():
    assert "disputed" in quality.assess(FULL, substate="not-applicable").flags
    assert "disputed" in quality.assess(FULL, substate="informative").flags
    assert "disputed" in quality.assess(FULL, labels=["HackerOne", "Not a vulnerability"]).flags
    assert "disputed" in quality.assess(FULL, comments=["Triage: this is working as designed and not a security issue."]).flags
    assert "disputed" not in quality.assess(FULL, substate="resolved", labels=["HackerOne", "Weakness::CWE-918"]).flags


def test_images_are_flagged_and_dependence_is_detected():
    imgs = [{"kind": "image", "ref": "a.png"}]
    assert "has_images" in quality.assess(FULL, images=imgs).flags and "image_dependent" not in quality.assess(FULL, images=imgs).flags
    flags = quality.assess("The bug is shown in the attached screenshot [image: a.png]. See the video for the steps.", images=imgs).flags
    assert {"has_images", "image_dependent"} <= set(flags)
    assert "image_dependent" in quality.assess(FULL + "\nFull exploit: see the attached video [video: poc.mp4]", images=[{"kind": "video", "ref": "poc.mp4"}]).flags


def test_ai_instruction_flags():
    assert "ai_instructions_removed" in quality.assess(FULL, injection_hits=["x"]).flags
    assert "ai_instructions_in_code" in quality.assess(FULL, injection_in_code=1).flags
    assert not ({"ai_instructions_removed", "ai_instructions_in_code"} & set(quality.assess(FULL).flags))


def test_score_is_bounded_and_penalises_image_dependence():
    s = quality.assess(FULL).score
    assert 0.0 <= s <= 1.0
    dep = quality.assess(FULL + "\nSee the attached video [video: poc.mp4]", images=[{"kind": "video", "ref": "poc.mp4"}]).score
    assert dep < s
