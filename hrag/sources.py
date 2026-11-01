"""The common shape every source (HackerOne API, GitLab copy) is converted to before cleaning."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SourceReport:
    source: str                      # 'hackerone_api' | 'gitlab'
    source_url: str                  # where this text really came from
    report_id: int                   # the HackerOne report id (the database key)
    title: str
    body: str                        # RAW text, cleaned later
    program: str = ""
    weakness: str = ""
    cwe: str = ""
    severity: str = ""
    substate: str = ""
    created_at: str = ""
    disclosed_at: str = ""
    labels: list[str] = field(default_factory=list)
    comments: list[dict] = field(default_factory=list)      # raw notes; filtered by clean.filter_comments
    notes: list[str] = field(default_factory=list)          # collection notes, e.g. "comments skipped: no GITLAB_TOKEN"
