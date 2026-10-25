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
