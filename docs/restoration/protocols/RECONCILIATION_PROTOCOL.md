# ZIP and Git Reconciliation Protocol

## Goal

Determine which project state is authoritative for each meaningful component.

## Evidence sources

1. ZIP file
2. local Git tree
3. Git history
4. remote Git tree
5. generated artifacts
6. notebooks
7. logs
8. documentation

## Comparison matrix

| Component | ZIP | Local Git | Remote Git | Historical evidence | Decision |
|---|---|---|---|---|---|
| Source | TBD | TBD | TBD | TBD | TBD |
| Config | TBD | TBD | TBD | TBD | TBD |
| Models | TBD | TBD | TBD | TBD | TBD |
| Data scripts | TBD | TBD | TBD | TBD | TBD |
| Notebooks | TBD | TBD | TBD | TBD | TBD |
| Docs | TBD | TBD | TBD | TBD | TBD |

## Classification

[ZIP_ONLY]
[REPO_ONLY]
[BOTH_SAME]
[BOTH_CONFLICT]
[HISTORICAL]
[GENERATED]
[UNKNOWN_PROVENANCE]

## Decision rule

Prefer the version with stronger evidence, not merely the newest timestamp.

Evidence strength:

1. reproducible execution
2. explicit commit provenance
3. dependency-consistent implementation
4. referenced artifact
5. test coverage
6. documentation claim
7. file timestamp alone

## Merge policy

Do not merge divergent implementations blindly.

For conflicts:

1. preserve both versions in evidence notes
2. identify behavioral difference
3. locate dependent code and artifacts
4. create a ticket
5. make an explicit decision
6. record the decision in DECISIONS.md
7. implement the chosen version
8. validate
