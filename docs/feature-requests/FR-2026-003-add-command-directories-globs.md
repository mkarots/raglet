# Feature Request: `add` Command Should Accept Directories and Glob Patterns

| Field | Value |
|---|---|
| **FR-ID** | FR-2026-003 |
| **Component** | `raglet.cli` — `add` command |
| **Priority** | Medium |
| **Type** | Enhancement |
| **Date** | 2026-03-08 |
| **Status** | Proposed |
| **Related** | — |

---

## Summary

The `add` command currently only accepts individual files. It should accept directories and glob patterns, consistent with how `build` already works.

---

## Problem Statement

`build` and `add` are conceptually the same operation — process inputs and embed them — except `build` starts from scratch and `add` appends to an existing raglet. But their input handling is inconsistent:

```bash
# build accepts all of these
raglet build src/ docs/ "*.md" --out .raglet/

# add only accepts individual files
raglet add --raglet .raglet/ file1.txt file2.txt  # works
raglet add --raglet .raglet/ src/                 # doesn't work
raglet add --raglet .raglet/ "*.md"               # doesn't work
```

This means users who added a new directory to their project, or want to incrementally index a folder of new documents, have to enumerate files manually. The fix is already in the codebase — `build` uses `expand_file_inputs()` from `raglet.utils`, and `add` just doesn't call it.

---

## Proposed Solution

Call `expand_file_inputs()` in `add_command()` before processing, identical to how `build_command()` does it. Also add `--ignore` and `--max-files` flags for consistency.

```bash
# After fix — all of these work
raglet add --raglet .raglet/ src/
raglet add --raglet .raglet/ docs/ "*.md"
raglet add --raglet .raglet/ new-docs/ --ignore "__pycache__"
```

---

## Acceptance Criteria

- `add` accepts directories, glob patterns, and individual files
- `add` supports `--ignore` (comma-separated patterns) consistent with `build`
- `add` supports `--max-files` consistent with `build`
- Behaviour is identical to `build` for input expansion
- All existing `add` tests pass
- New tests cover directory and glob inputs

---

## Risks

Low. The input expansion logic is already tested via `build`. This is a small plumbing change.
