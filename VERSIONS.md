# VERSIONS

Each change of the crates' version and the public API changes it carries
(AGENTS.md, Rule 5). Internal changes are not listed, and there is no entry
without a version change. Entries are appended, oldest first, and never
edited.

## 0.1.0 — 2026-10-25 (`acb8f8e`, tag `v0.1.0`)

Initial release, extracted from flankr (`crates/libinfer`).

`libinfer`:
- `ReasoningEngine`, `ToolExecutor`, `TurnContributor` async traits.
- `Message`, `Role` conversation data types.
- `SessionManager`, `SessionBuilder` JSONL session handling.
- `RoutingReasoningEngine`, `Route`.
- `MockReasoningEngine`, `MockToolExecutor`, `MockHandler`.

`libinfer-gemini`:
- `GeminiReasoningEngine`, `GeminiReasoningEngineBuilder`.
- Re-exports of `gemini-rust` 1.7 types (`Content`, `FunctionCall`, `FunctionCallingMode`, `FunctionDeclaration`, `Gemini`, `Message`, `Model`, `Role`, `Tool`).

## 0.1.1 — 2026-10-30 (`f2d8d36`, tag `v0.1.1`)

`libinfer-gemini` only (`libinfer` stays 0.1.0). Strictly additive: a
Gemini model asked for by tier and age, resolved against the key's model
list (docs/design/20261030_01_GEMINI_MODEL_CHOICE_PLAN.md).

`libinfer-gemini`:
- Added `Tier` (`FlashLite`, `Flash`, `Pro`) and `Age` (`Newest`, `Oldest`), both `#[non_exhaustive]`, with `Display`.
- Added `ModelChoice` (`Rule { tier, age }`, `Named(String)`; `#[non_exhaustive]`), with `ModelChoice::parse` and `Display` for the text form `<tier>:<age>` (`flash-lite:oldest`).
- Added `ModelCatalog`, with `fetch` (the key's `generateContent` models, every page), `from_names`, `resolve` and `names`.
- Added `Resolved` (`model`, `how`; `#[non_exhaustive]`) and `How` (`Alias`, `Matched`, `Named`, `NamedMissing`, `Fallback`; `#[non_exhaustive]`).
- Added `resolve_offline`.
- `DEFAULT_MODEL` and `model_from_name` are unchanged.

## 0.2.0 — 2026-10-30 (`3641ed2`, tag `v0.2.0`)

`libinfer-gemini` only (`libinfer` stays 0.1.0). A floor for *oldest*
(docs/design/20261030_01_GEMINI_MODEL_CHOICE_PLAN.md, the 2026-10-30
15:23 EDT amendment).

`libinfer-gemini`:
- **Breaking (behaviour):** `ModelCatalog::resolve` for `Age::Oldest` picks the lowest stable version at or above `OLDEST_FLOOR` (3.0); versions below it never count, and a tier with none at or above the floor resolves to `DEFAULT_MODEL` (`How::Fallback`). `Age::Newest` and `ModelChoice::Named` are unchanged.
- Added `OLDEST_FLOOR: (u64, u64)`, the floor as `(major, minor)`.
