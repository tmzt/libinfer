# libinfer

`libinfer` is an asynchronous Rust library providing a provider-neutral reasoning layer and LLM execution abstractions.

The workspace consists of:
- `libinfer`: The core traits (`ReasoningEngine`, `ToolExecutor`, `TurnContributor`), universal conversation types (`Message`, `Role`), JSONL session persistence, and deterministic mock engines for testing.
- `libinfer-gemini`: Gemini reasoning engine adapter built directly on `gemini-rust`.

## Architectural decisions

| Document | Date | Status | Decision |
| --- | --- | --- | --- |
| None | 2026-10-25 | Implemented | Initial repository extraction from flankr. |
