# libinfer

`libinfer` is an asynchronous Rust library providing a provider-neutral reasoning layer and LLM execution abstractions.

The workspace consists of:
- `libinfer`: The core traits (`ReasoningEngine`, `ToolExecutor`, `TurnContributor`), universal conversation types (`Message`, `Role`), JSONL session persistence, and deterministic mock engines for testing.
- `libinfer-gemini`: Gemini reasoning engine adapter built directly on `gemini-rust`.

## Architectural decisions

| Document | Date | Status | Decision |
| --- | --- | --- | --- |
| None | 2026-10-25 | Implemented | Initial repository extraction from flankr. |
| [Gemini model choice by tier and age](docs/design/20261030_01_GEMINI_MODEL_CHOICE_PLAN.md) | 2026-10-30 | Implemented | A model asked for by tier (Flash-Lite, Flash, Pro) and age (newest or oldest served), resolved once against the key's model list; built in libinfer-gemini 0.1.1, with a 3.0 floor on oldest in 0.2.0. |
