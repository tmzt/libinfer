# CHANGES

Planned new work, feature plans and architectural changes, appended in order
(AGENTS.md, Rule 0). Entries are never edited.

## 2026-10-25 14:53 EDT — Repository creation and extraction from flankr

- `libinfer` is extracted from `flankr` into its own standalone workspace repository.
- Contains `libinfer` (provider-neutral core traits, session manager, and mock engines)
  and `libinfer-gemini` (Gemini provider on `gemini-rust`).
