# Ticketing Protocol

## Objective

Convert discovery into a structured, searchable engineering backlog.

## One issue per independently actionable problem

Create separate tickets when fixes can be developed, tested or validated independently.

Create separate issues for:

- dependency breakage
- stale imports
- broken data paths
- model interface mismatches
- missing tests
- contradictory documentation
- irreproducible experiments
- missing result artifacts
- architecture problems
- security problems
- performance problems

## Ticket structure

### Title

`[TYPE] [PRIORITY] concise problem`

Example:

`[BUG] [P1] stage3 gate loader cannot restore saved checkpoint`

### Body

```text
## Problem

What is wrong?

## Evidence

What files, commands, logs, commits or artifacts prove the problem?

## Impact

Why does it matter?

## Suspected cause

What currently appears to be the cause?

## Scope

What should this ticket change?

## Acceptance criteria

- [ ] ...
- [ ] ...

## Validation

Exact command or test that proves resolution.

## Dependencies

Related tickets or decisions.

## Documentation

Documents that must change when the ticket closes.
```

## Status discipline

Never move directly from OPEN to CLOSED.

Preferred flow:

OPEN -> INVESTIGATING -> READY -> IN_PROGRESS -> VERIFY -> CLOSED

Use BLOCKED when external information, data, hardware or another ticket prevents progress.

## Ticket linking

Every implementation commit should reference the ticket.

Every closed ticket should reference:

- implementation commit
- validation evidence
- changed documents

Every important document should reference relevant tickets.

## Multi-ticket discovery

When one inspection uncovers several problems, create several tickets in one pass.

Example:

- [BUG] stale import
- [DATA] undocumented preprocessing
- [TEST] missing regression test
- [DOC] README claims unsupported behavior
- [REPRO] missing environment pin

Do not collapse these into "project cleanup".

## Ticket quality test

A good ticket allows another engineer to begin work without asking the discoverer what they meant.
