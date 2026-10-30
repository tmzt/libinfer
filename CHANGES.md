# CHANGES

Planned new work, feature plans and architectural changes, appended in order
(AGENTS.md, Rule 0). Entries are never edited.

## 2026-10-25 14:53 EDT — Repository creation and extraction from flankr

- `libinfer` is extracted from `flankr` into its own standalone workspace repository.
- Contains `libinfer` (provider-neutral core traits, session manager, and mock engines)
  and `libinfer-gemini` (Gemini provider on `gemini-rust`).

## 2026-10-30 13:49 EDT — Gemini model choice by tier and age

Plan: `docs/design/20261030_01_GEMINI_MODEL_CHOICE_PLAN.md` (Draft). On the
owner's instruction (2026-10-30), after flankr's summaries failed on a
Flash-Lite name the API never served: libinfer-gemini gains `Tier`, `Age`
and `ModelChoice`, a `ModelCatalog` fetched from the API's model list, and
a fuzzy match that resolves a choice to a served model, falling back to
`DEFAULT_MODEL`. Additive (0.1.1).
