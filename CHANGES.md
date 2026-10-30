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

## 2026-10-30 13:59 EDT — Gemini model choice: small departures in the build

Plan: `docs/design/20261030_01_GEMINI_MODEL_CHOICE_PLAN.md`, steps 1–4.

- Step 1: an empty `ModelCatalog` (`ModelCatalog::default()`, or a
  fetched list with no `generateContent` models) counts as no list and
  resolves as `resolve_offline`, so a `Named` choice gives `How::Named`
  rather than `NamedMissing` (the "(or no list to check against)" case of
  `How::Named`). Rules fall back to `DEFAULT_MODEL` either way.
- Step 2: each page request has a 30-second timeout, and an HTTP error
  keeps at most the first 300 characters of the response body. Any
  occurrence of the key in an error's text is replaced with `<key>`.
- Step 3: `resolve_offline` logs as `resolve` does (decision 6 asks for
  one line per resolution): an `info` for a name, a `warn` naming "no
  model list" for a rule's fallback.
- Step 4: the `VERSIONS.md` entry names `f2d8d36`, the last code
  commit, as the 0.1.0 entry named the last code commit; the tag
  `v0.1.1` goes on the bump commit.
