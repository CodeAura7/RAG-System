"""Quality flags and a 0..1 score for a cleaned report. The flags are stored with the report and with every chunk; the score
is only used to prefer better reports when selecting which ~100 to keep. Nothing here judges whether a bug is real."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from .chunking import classify_section

_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$", re.M)
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+\S", re.M)
_HTTP = re.compile(r"^\s*(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+/\S*", re.M)
_IMPACT_SENTENCE = re.compile(r"\b(an? )?attackers? (can|could|may|would)\b|\ballows? (an? )?(attacker|remote|unauthenticated)|\bimpact\b", re.I)
_IMAGE_DEPENDENCE = re.compile(
    r"\b(see|as (?:shown|seen)|attached|watch|refer to|check|in the|shown in)\b[^.\n]{0,40}\b(screenshots?|videos?|gif|recording|images?|pictures?)\b", re.I)
_DISPUTE_SUBSTATE = {"not-applicable", "informative", "spam", "duplicate", "wontfix", "invalid", "not-a-bug"}
_DISPUTE_TEXT = re.compile(
    r"not an? (?:security )?(?:vulnerability|issue|bug)|(?:works?|working) as (?:designed|intended)|\bby design\b|intended behou?r|"
    r"out of scope|(?:cannot|can't|unable to|could not) reproduce|false positive|won'?t fix|\bwontfix\b|\binformative\b|not applicable|\bduplicate of\b",
    re.I)


@dataclass
class Assessment:
    flags: list[str] = field(default_factory=list)
    score: float = 0.0
    has_steps: bool = False
    has_poc: bool = False
    has_impact: bool = False


def assess(text: str, *, images: Sequence[dict] = (), injection_hits: Sequence[str] = (), injection_in_code: int = 0,
           substate: str = "", labels: Sequence[str] = (), comments: Sequence[str] = (), sparse_chars: int = 600) -> Assessment:
    sections = {classify_section(h) for h in _HEADING.findall(text)}
    has_steps = "steps" in sections or len(_NUMBERED.findall(text)) >= 2
    has_poc = "```" in text or "~~~" in text or bool(_HTTP.search(text)) or "curl " in text
    has_impact = "impact" in sections or bool(_IMPACT_SENTENCE.search(text))
    sparse = len(text.strip()) < sparse_chars or len(text.split()) < 80
    images = list(images)
    image_dependent = bool(images) and (sparse or not has_poc or bool(_IMAGE_DEPENDENCE.search(text)))
    disputed = (substate.lower() in _DISPUTE_SUBSTATE
                or any(_DISPUTE_TEXT.search(label.replace("-", " ").replace("_", " ")) for label in labels)
                or any(_DISPUTE_TEXT.search(c) for c in comments))

    flags: list[str] = []
    if sparse:
        flags.append("sparse")
    if not has_steps:
        flags.append("no_steps")
    if not has_poc:
        flags.append("no_poc")
    if not has_impact:
        flags.append("no_impact")
    if not has_steps and not has_poc:
        flags.append("incomplete")
    if disputed:
        flags.append("disputed")
    if images:
        flags.append("has_images")
    if image_dependent:
        flags.append("image_dependent")
    if injection_hits:
        flags.append("ai_instructions_removed")
    if injection_in_code:
        flags.append("ai_instructions_in_code")

    # steps + PoC + impact carry 70%; length (saturating at 1500 characters) 15%; not being disputed 15%
    score = (0.25 * has_steps + 0.25 * has_poc + 0.20 * has_impact + 0.15 * min(len(text) / 1500, 1.0) + 0.15 * (not disputed))
    score -= 0.15 * image_dependent + 0.20 * sparse
    return Assessment(flags, round(max(0.0, min(1.0, score)), 3), has_steps, has_poc, has_impact)
