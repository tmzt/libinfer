---
title: Gemini model choice by tier and age, resolved against the account's models
date: 2026-10-30
status: Implemented
covers:
  - "A `ModelChoice` in libinfer-gemini: a model asked for by its tier
    (Flash-Lite, Flash, Pro) and its age (newest or oldest still served),
    or by an exact name as today"
  - "Fetching the account's model list from the Gemini API once, and
    resolving a choice against it by a fuzzy match on the model names"
  - "The text form a config uses (`flash-lite:oldest`), the fallback when
    nothing matches, and what is logged"
  - "Not covered: changing `DEFAULT_MODEL` or `model_from_name`; other
    providers; flankr's and RoundTwo's adoption, each their own amendment"
sources:
  - "Owner (2026-10-30, verbatim): \"What I'd like to do is to have an
    enom that has something like Gemini Oldest, Gemini Oldest Flashlight,
    Gemini Oldest Flash, Gemini Newest, no, the same combination.\""
  - "Owner (2026-10-30, verbatim): \"the version. What I mean is, if it
    matches what Gemini API does, the newest, I think the newest can just
    emit the version number. So like we had, like Gemini Flash would match
    automatically on the latest version. I'm not sure as if Num should be
    split into two enoms. So the level and the, like the oldest, newest,
    basically what I'm looking for is a fuzzy match so that it optimizes
    between, you know, the newest, which would have, which has a higher
    cost and the oldest, which is a lower cost. And this would be a plan
    within [libinfer]\""
  - "flankr's daemon log, 2026-10-27 to 2026-10-30: 336 summary requests
    failed with `models/gemini-3.8-flash-lite is not found for API version
    v1beta`; flankr's summary default named a model that was never served
    to the key"
related:
  - "flankr docs/design/20261022_07_LIBINFER_PROVIDERS_PLAN.md (libinfer-gemini
    on gemini-rust directly)"
  - "flankr docs/design/20261027_02 (the `[llm]` config; its model defaults
    adopt this plan by a later amendment)"
  - "RoundTwo docs/design/20261029_01_INTERVIEW_COACH_PLAN.md, the model
    decision (prompts without a model use `DEFAULT_MODEL`)"
---

# Gemini model choice by tier and age

## 1. Summary

Callers name Gemini models by hand today (`gemini-3.8-flash`,
`gemini-3.8-flash-lite`). A name that is retired, or was never served to
the key, fails every request: flankr's summaries failed 336 times on a
Flash-Lite name that does not exist. This plan lets a caller ask for a
model by **tier** and **age** instead, for example *the oldest Flash-Lite
still served* (cheapest) or *the newest Flash* (best, costs more), and
resolves that once against the models the account's key can actually use.
Exact names keep working as they do now.

## 2. Findings (read 2026-10-30)

1. `libinfer-gemini` 0.1.0 (`libinfer-gemini/src/lib.rs`) has
   `DEFAULT_MODEL = "gemini-3.8-flash"` and `model_from_name(&str) ->
   Model`, which maps a few known names to `gemini_rust::Model` variants
   and passes any other through as `Model::Custom("models/<name>")`. The
   builder takes a name (`GeminiReasoningEngineBuilder::model`).
2. `gemini-rust` 1.7.1 has no call to list models. It sends requests with
   the `x-goog-api-key` header over `reqwest` 0.12.
3. The Gemini API lists models at
   `GET https://generativelanguage.googleapis.com/v1beta/models`, with the
   key in `x-goog-api-key`, paged by `pageSize` (up to 1000) and
   `pageToken`. Each entry has `name` (`models/gemini-2.5-flash`),
   `displayName`, `version` and `supportedGenerationMethods` (text models
   include `generateContent`); the page carries `nextPageToken` when there
   is more.
4. The API also serves version-free aliases for the newest of a family,
   such as `gemini-flash-latest` and `gemini-flash-lite-latest`, when they
   are in the list. A name without a version that the API serves moves
   with the API, which is what the owner asked for in *newest*.
5. Callers: flankr names its own models in `src/llm/gemini.rs`
   (`GENERAL_AGENT_MODEL`, `SUMMARY_MODEL`); RoundTwo uses
   `DEFAULT_MODEL` when a prompt names none.

## 3. Decisions

1. **Two enums, not one.** The tier and the age vary independently, so
   they are two enums, and a choice is either a rule over them or an exact
   name:

   ```rust
   /// A Gemini model's level.
   #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
   #[non_exhaustive]
   pub enum Tier { FlashLite, Flash, Pro }

   /// Which of a tier's served versions: the newest (the best, at a higher
   /// cost) or the oldest still served (the cheapest).
   #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
   #[non_exhaustive]
   pub enum Age { Newest, Oldest }

   /// A model asked for.
   #[derive(Clone, Debug, PartialEq, Eq)]
   #[non_exhaustive]
   pub enum ModelChoice {
       /// The newest or oldest served model of a tier.
       Rule { tier: Tier, age: Age },
       /// An exact model name, used as given (`gemini-2.5-flash`).
       Named(String),
   }
   ```

2. **The text form,** for configs: `<tier>:<age>`, with the tier
   `flash-lite`, `flash` or `pro` and the age `newest` or `oldest`, case
   ignored and spaces trimmed (`flash-lite:oldest`, `flash:newest`). A
   leading `gemini-` is allowed (`gemini-flash:newest`). Anything else is
   `Named`. `ModelChoice::parse(&str) -> ModelChoice` never fails;
   `Display` gives the text form back.

3. **The match.** A listed model counts only if its
   `supportedGenerationMethods` has `generateContent`. Its name, without
   `models/`, is read as `gemini-<version>-<tier>[-<suffix>]`:
   - `<version>` is digits with an optional `.` and digits (`2`, `2.5`,
     `3.8`), compared numerically (major, then minor; a missing minor is
     0);
   - `<tier>` is the longest of `flash-lite`, `flash`, `pro` that
     matches, so `flash-lite` is never read as `flash`;
   - `<suffix>`: none, or a digits-only revision (`-001`), is **stable**
     and counts. Any other suffix (`-preview…`, `-exp…`, `-tts`, `-image…`,
     `-live…`, `-native-audio…`, `-thinking…`) does not count for a rule.
   - Names not in this shape (embeddings, `gemma-…`, `learnlm-…`) never
     match a rule.
   - The version-free aliases `gemini-<tier>-latest` are kept apart for
     *newest* (item 4).

4. **Resolving a rule:**
   - **Newest:** `gemini-<tier>-latest` if the list has it; otherwise the
     stable model of that tier with the highest version, and among the same
     version the one without a revision suffix, then the highest revision.
   - **Oldest:** the stable model of that tier with the lowest version,
     and among the same version the one without a revision suffix, then the
     lowest revision. When Google retires it, the next one up is chosen
     with no code change.
   - **Named:** used as given; when a list is known and does not have it,
     one warning is logged (item 6), and the name is still used.

5. **The fallback.** With no list (the fetch failed, or the caller has
   none), or no model of the tier in the list, a rule resolves to
   `DEFAULT_MODEL`. `DEFAULT_MODEL` and `model_from_name` do not change.

6. **What is logged,** with `log`: one `info` line per resolution, naming
   the choice and the model (`gemini: flash-lite:oldest is
   gemini-2.5-flash-lite`); one `warn` for a fallback or a named model the
   list lacks, naming the choice, why, and the model used. The key is never
   logged.

7. **Fetched once, by the caller.** `ModelCatalog::fetch` makes the
   request (all pages) once; the caller keeps the catalog for the life of
   its process and resolves each of its choices against it. Nothing is
   written to disk, and there is no background refresh: a process started
   after a model is retired resolves afresh.

8. **Cost** is not read from the API (the list has no prices). *Newest*
   and *oldest* stand for "higher cost, better" and "lower cost" as the
   owner put it; the doc comments say so.

## 4. Interfaces (libinfer-gemini)

```rust
/// The models an account's key can use for text, as the API listed them.
#[derive(Clone, Debug, Default)]
pub struct ModelCatalog { /* the listed names that support generateContent */ }

impl ModelCatalog {
    /// Lists the key's models: `GET …/v1beta/models?pageSize=1000`, every
    /// page, the key in `x-goog-api-key`. Errors are the HTTP or JSON
    /// failure as text, never containing the key.
    pub async fn fetch(api_key: &str) -> Result<ModelCatalog, String>;

    /// A catalog of these names (each with or without `models/`), all taken
    /// as supporting `generateContent`: for tests and offline callers.
    pub fn from_names<I, S>(names: I) -> ModelCatalog
    where I: IntoIterator<Item = S>, S: AsRef<str>;

    /// The model name `choice` resolves to (without `models/`), and how.
    pub fn resolve(&self, choice: &ModelChoice) -> Resolved;

    /// The names listed, sorted.
    pub fn names(&self) -> Vec<String>;
}

/// What a choice resolved to.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Resolved {
    pub model: String,
    pub how: How,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum How {
    /// The tier's `-latest` alias.
    Alias,
    /// A versioned model the rule picked.
    Matched,
    /// The exact name given, in the list (or no list to check against).
    Named,
    /// The exact name given, missing from the list.
    NamedMissing,
    /// Nothing matched: `DEFAULT_MODEL`.
    Fallback,
}

/// Resolves `choice` without a catalog: a `Named` choice as given, a rule
/// to `DEFAULT_MODEL` (decision 5).
pub fn resolve_offline(choice: &ModelChoice) -> Resolved;
```

The resolved `model` goes to `GeminiReasoningEngineBuilder::model` as
today. The HTTP request uses `reqwest` 0.12 as `gemini-rust` does, with
the same TLS features, so no second TLS stack is added; it runs under the
crate's existing `async-compat` use.

## 5. Steps

1. **The types and the match.** `Tier`, `Age`, `ModelChoice` (with
   `parse` and `Display`), `ModelCatalog::from_names`, `resolve`,
   `Resolved`, `How` and `resolve_offline` in a new
   `libinfer-gemini/src/choice.rs`, re-exported from the crate root.
   **Done when:** unit tests pass for: each tier and age over a list such
   as `gemini-2.0-flash`, `gemini-2.0-flash-lite`, `gemini-2.5-flash`,
   `gemini-2.5-flash-lite`, `gemini-2.5-pro`, `gemini-3.8-flash`,
   `gemini-3.8-flash-preview`, `gemini-2.5-flash-001`,
   `gemini-flash-latest`, `text-embedding-004`; `flash-lite` never read as
   `flash`; previews and other suffixes skipped; the alias preferred for
   *newest*; a tier with no model falling back to `DEFAULT_MODEL`; a named
   model present and missing; the text form round trip, case and spaces.
2. **Fetching the list.** `ModelCatalog::fetch` with paging, against a
   local test server (no real API call in tests), parsing the documented
   fields and keeping only `generateContent` models. **Done when:** tests
   cover two pages, a model without `generateContent` left out, an HTTP
   error and bad JSON given as errors without the key in their text.
3. **Logging.** `resolve` logs as decision 6. **Done when:** a test with a
   captured logger sees one `info` for a match and one `warn` for a
   fallback and for a missing named model, and no key.
4. **The release.** libinfer-gemini's public API grows only by new items
   (strictly additive), so the version goes to 0.1.1, with its
   `VERSIONS.md` entry in the same commit and the tag `v0.1.1`.
   **Done when:** `cargo fmt --check`, `cargo clippy --all-targets` (no new
   warnings) and `cargo test` pass for the workspace, and the tag is on the
   bump commit.

## 6. Adoption (outside this plan)

- **flankr:** an amendment to its `[llm]` plan: the daemon fetches the
  catalog at start; summaries default to `flash-lite:oldest`, the general
  agent and planning to `flash:newest`; a config `model` takes either the
  text form or an exact name; the names chosen are logged at start.
- **RoundTwo:** optional; its prompts could name `flash:newest` instead of
  relying on `DEFAULT_MODEL`.

## 7. Open questions for the owner

1. **Pro:** keep `Pro` as a tier, so a config could ask for
   `pro:newest`, or leave it out until something needs it? (flankr's
   `[llm]` plan says never Pro unless a config names one; the tier would
   not change that, since no default would use it.)
2. **Previews for newest:** should *newest* ever pick a `-preview` model
   when the tier has no stable one, or always fall back?
3. **Oldest and quality:** should *oldest* have a floor (for example
   never below version 2.0), so a very old model left in the list is not
   chosen?

## Amendment — 2026-10-30 13:53 EDT: the open questions answered

The owner (2026-10-30) agreed to the proposed answers to section 7:

1. **Pro stays a tier.** `Tier::Pro` is kept, so a config may ask for
   `pro:newest` or `pro:oldest`; no default uses it.
2. **Newest never picks a preview.** A `-preview` (or any other
   non-stable suffix) never matches a rule, as decision 3 says; a tier
   with no stable model and no `-latest` alias falls back to
   `DEFAULT_MODEL` (decision 5).
3. **No version floor for oldest.** *Oldest* is the lowest stable version
   listed, whatever it is.

Section 7 has no open questions left; the rest of the plan is unchanged.

## Amendment — 2026-10-30 13:53 EDT: approved

The owner approved the plan as it stands at commit `527a838` (2026-10-30:
"Approve plan 527a838"). The status stays `Draft` until its first
implementing commit sets `In implementation` (Rule 1.5).

## Amendment — 2026-10-30 13:56 EDT: implementation started; the document is now immutable

Implementation started on the owner's approval at `527a838`, with step 1
(the types and the match). The status is `In implementation`, and the
document is now immutable (Rule 1.5).

## Amendment — 2026-10-30 13:58 EDT: implemented

Every step is done, and the status is `Implemented` (Rule 1.3). The
commits, on the `model-choice` branch:

1. **The types and the match:** `11b35e6` (`libinfer-gemini/src/choice.rs`,
   re-exported from the crate root; this commit also set
   `In implementation`).
2. **Fetching the list:** `bdfb5fa` (`ModelCatalog::fetch`, with
   `reqwest` 0.12 on `rustls-tls` only, the TLS `gemini-rust` 1.7.1
   enables; no new packages in `Cargo.lock`).
3. **Logging:** `f2d8d36`.
4. **The release:** `ec249c5` (libinfer-gemini 0.1.1 and its
   `VERSIONS.md` entry). The tag `v0.1.1` goes on `ec249c5` when the
   branch lands; the owner tags it.

## Amendment — 2026-10-30 15:23 EDT: a floor for *oldest*: the first 3.x model

On the owner's instruction (2026-10-30): "right, set a floor on the first
3.x model, we could add a ForceOldest (assuming the user has grandfathered
access) but that might be too much". The reason: a model can be in the
key's list and still be refused. flankr's summaries resolved
`flash-lite:oldest` to `gemini-2.5-flash-lite`, which the API listed but
refused: "This model models/gemini-2.5-flash-lite is no longer available
to new users. Please update your code to use models/gemini-3.5-flash-lite".
This replaces the 13:53 EDT amendment's answer 3 ("no version floor for
oldest") and adds to decision 4:

1. **The floor.** `OLDEST_FLOOR` is version 3.0. *Oldest* picks the
   stable model of the tier with the lowest version **at or above** the
   floor, by decision 4's ordering. Versions below it never count for
   *oldest*.
2. **Nothing at or above the floor** in the tier: *oldest* falls back to
   `DEFAULT_MODEL` (decision 5), with the usual warning naming the
   choice.
3. **Newest and named choices** are unchanged. A `Named` model below the
   floor is still used as given.
4. **No ForceOldest.** An age that ignores the floor, for keys with older
   access, is left out for now.
5. **Release.** This changes what `ModelCatalog::resolve` returns for
   *oldest*, behaviour a caller observes, so it is not strictly additive
   (Rule 5.4): libinfer-gemini 0.2.0, its `VERSIONS.md` entry marked
   **Breaking (behaviour)**, tag `v0.2.0`. `OLDEST_FLOOR` is public.

## Amendment — 2026-10-30 15:26 EDT: back in implementation for the floor; the document is immutable again

The status returns from `Implemented` to `In implementation` for the
2026-10-30 15:23 EDT amendment (a floor for *oldest*: the first 3.x
model). The document stays immutable; it returns to `Implemented` when
that amendment is built.

## Amendment — 2026-10-30 15:26 EDT: implemented again

The 2026-10-30 15:23 EDT amendment (a floor for *oldest*) is built; the
status returns to `Implemented`. The commits, on the `oldest-floor`
branch:

1. **The floor** (items 1 to 3): `3641ed2` (`OLDEST_FLOOR` and
   `Age::Oldest` in `libinfer-gemini/src/choice.rs`, re-exported from
   the crate root; this commit also set `In implementation`).
2. **The release** (item 5): `2034383` (libinfer-gemini 0.2.0 and its
   `VERSIONS.md` entry, marked **Breaking (behaviour)**). The tag
   `v0.2.0` goes on `2034383` when the branch lands; the owner tags it.

## Amendment — 2026-10-30 15:37 EDT: a refused model is replaced by the policy's next choice

On the owner's instruction (2026-10-30), after the floor: "yes, it should
bump the version selection when it gets that error, if that can happen in
the initial call or on the first actual call without requiring additional
client code"; then, told that stepping by our own order needs the engine to
hold the choice and the list: "it should validate Google's suggestion
against the policy (if it pushes 3.8 but 3.6 is in the list, it should use
3.6)"; then "Yes, add the floor check to the libinfer plan". A model can be
listed and still refused ("This model models/<name> is no longer available
to new users. Please update your code to use models/<suggested>"). This
adds:

1. **The builder takes a choice.** `GeminiReasoningEngineBuilder::choice(
   choice: ModelChoice, catalog: &ModelCatalog) -> Self` (additive). It
   resolves the choice against the catalog for the first model, as
   `ModelCatalog::resolve` does, and keeps the choice and a copy of the
   catalog in the engine. `model(name)` is unchanged and never switches.
2. **A refusal.** When a request answers HTTP 404 with status `NOT_FOUND`
   naming the engine's model, the engine marks that model refused (for its
   life) and picks again by the policy, without the refused models:
   - **a rule** (`Rule { tier, age }`): the next model the same rule picks
     from the catalog with the refused ones left out: for *oldest*, the
     next lowest stable version at or above `OLDEST_FLOOR`; for *newest*,
     the next highest (after the `-latest` alias, if that was refused);
   - **Google's suggestion** (the `models/<name>` after "use" in the
     message) is taken only when the rule has nothing left, and only if it
     is the same tier, stable, and, for *oldest*, at or above the floor;
   - **a named choice** never switches: the error is returned as now.
3. **Retried once per switch,** at most 3 switches for the engine's life;
   the request is sent again on the new model, and the caller sees its
   answer. With nothing left, the 404 is returned as now.
4. **Logged:** one `warn` per switch, naming the refused model, the new
   one and how it was chosen (`policy` or `suggested`), never the key.
5. **Release.** Additive (a new builder method; `model()` and every
   existing behaviour unchanged): libinfer-gemini 0.2.1, tag `v0.2.1`.
6. **Adoption** (flankr, its own amendment): the daemon builds its clients
   with `.choice(choice, &catalog)` where it now passes the resolved name.
7. **Tests** against the local test server: a refused *oldest* model steps
   to the next 3.x one, not to Google's suggestion when the list has a
   lower one (the owner's case: suggested 3.8, 3.6 listed, 3.6 used);
   Google's suggestion used when the rule has nothing left, and refused
   when below the floor or another tier; a named model not switched; at
   most 3 switches; the key in no log line.

## Amendment — 2026-10-30 15:39 EDT: the refused-model addition approved

The owner approved the amendment before this one, as committed at
`ea79d38` (2026-10-30: "Yes, approve the refused-model plan addition and
build it"). It is built now; the first implementing commit reopens the
plan (`In implementation`), as the floor's did.
