# Project Restoration Rules

## Rule 1: Evidence before edits

The first job is reconstruction, not coding.

## Rule 2: ZIP and repository are both evidence

Neither is automatically authoritative.

## Rule 3: Preserve historical intent

Historical files may contain useful implementations, experiments or design rationale. Do not delete them merely because they are old.

## Rule 4: Make uncertainty visible

Use [UNKNOWN], [UNVERIFIED] or [CLAIMED] when evidence is insufficient.

## Rule 5: Ticket independent defects separately

Several defects should become several tickets. Do not hide a backlog inside one omnibus issue.

## Rule 6: Keep commits scoped

A commit should answer one engineering question.

Good:
refactor data manifest loading
fix stage3 gate checkpoint loading
add regression tests for speaker counting

Bad:
fix everything
project cleanup
restore project

## Rule 7: Document decisions, not just actions

"Changed X to Y" is not enough. Record why X was inadequate, why Y was selected and what tradeoff exists.

## Rule 8: No silent behavioral change

Any meaningful behavior change needs a test, experiment or explicit decision record.

## Rule 9: No historical Git rewriting by default

Historical author records are evidence of the repository history. Do not rewrite them merely for appearance.

New commits must use the configured Rishi Garg identity and must contain no co-author or agent attribution.

## Rule 10: Push only after local verification

Never push a commit that has not passed the appropriate validation level.

## Rule 11: Documentation is executable knowledge

Documentation must correspond to actual code and verified evidence.

## Rule 12: Reproducibility is part of correctness

A model result that cannot be traced to a data recipe, code version and configuration is not a reproducible result.

## Rule 13: Remove nothing without provenance review

Before deleting old code, classify it as:

[DEAD]
[DUPLICATE]
[SUPERSEDED]
[UNSAFE]
[UNUSED]
[UNKNOWN]

Delete [UNKNOWN] only after investigation.

## Rule 14: Keep generated artifacts distinguishable

Never make generated outputs look like source code.

## Rule 15: Protect credentials

Do not commit secrets, tokens, cookies, private keys or local credential files.
