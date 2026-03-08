# Feature Requests

This directory tracks proposed and accepted feature requests for raglet.

## Requests

| ID | Title | Priority | Status |
|----|-------|----------|--------|
| [FR-2026-001](FR-2026-001-remove-embeddings-cache.md) | Remove redundant `self.embeddings` cache | High | Accepted |
| [FR-2026-002](FR-2026-002-simplify-storage-backends.md) | Clarify storage backend roles | Medium | Proposed |
| [FR-2026-003](FR-2026-003-add-command-directories-globs.md) | `add` command: directories and globs | Medium | Proposed |

## Format

Each feature request uses this structure:

| Field | Description |
|-------|-------------|
| **FR-ID** | Unique identifier (`FR-YYYY-NNN`) |
| **Component** | Module or area affected |
| **Priority** | High / Medium / Low |
| **Type** | Enhancement, Performance, Simplification, etc. |
| **Date** | When the request was filed |
| **Status** | Proposed, Accepted, Implemented, or Rejected |
| **Related** | Links to ADRs, other FRs, or issues |

Body sections: **Summary**, **Motivation**, **Proposal**, and **Acceptance Criteria**.

## Adding a New Feature Request

1. Create `FR-YYYY-NNN-short-title.md` using the next sequential number
2. Follow the format above
3. Update this README table
4. If accepted, link the corresponding ADR or implementation PR
