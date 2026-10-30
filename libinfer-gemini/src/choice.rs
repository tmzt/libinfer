//! A Gemini model asked for by its tier and age, or by an exact name, and
//! resolved against the models an account's key can use
//! (docs/design/20261030_01_GEMINI_MODEL_CHOICE_PLAN.md).
//!
//! The Gemini API lists no prices, so *newest* and *oldest* stand for
//! "the best, at a higher cost" and "the cheapest" (decision 8).

use std::collections::BTreeSet;
use std::fmt;
use std::time::Duration;

use async_compat::CompatExt;
use serde::Deserialize;

use crate::DEFAULT_MODEL;

/// Where the Gemini API lists models.
const MODELS_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// The largest page the API serves.
const PAGE_SIZE: &str = "1000";

/// How long one page request may take.
const FETCH_TIMEOUT: Duration = Duration::from_secs(30);

/// The most of an error response's body kept in an error.
const ERROR_BODY_CHARS: usize = 300;

/// The lowest version *oldest* may pick, as `(major, minor)`: 3.0, the
/// first 3.x model. A model can be in the key's list and still be refused
/// ("no longer available to new users"), as `gemini-2.5-flash-lite` was, so
/// versions below this never count for [`Age::Oldest`]. [`Age::Newest`] and
/// [`ModelChoice::Named`] do not use it: a named model below the floor is
/// used as given.
pub const OLDEST_FLOOR: (u64, u64) = (3, 0);

/// A Gemini model's level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Tier {
    /// `flash-lite`: the cheapest and fastest.
    FlashLite,
    /// `flash`.
    Flash,
    /// `pro`: the most capable, and the most costly.
    Pro,
}

impl Tier {
    /// The tier as it appears in model names and the text form.
    fn as_str(self) -> &'static str {
        match self {
            Tier::FlashLite => "flash-lite",
            Tier::Flash => "flash",
            Tier::Pro => "pro",
        }
    }

    fn from_text(text: &str) -> Option<Tier> {
        match text {
            "flash-lite" => Some(Tier::FlashLite),
            "flash" => Some(Tier::Flash),
            "pro" => Some(Tier::Pro),
            _ => None,
        }
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Which of a tier's served versions: the newest (the best, at a higher
/// cost) or the oldest still served (the cheapest).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Age {
    /// The newest served version: the best, at a higher cost.
    Newest,
    /// The oldest stable version served at or above [`OLDEST_FLOOR`]
    /// (3.0): the cheapest that new keys are still given. With none at or
    /// above the floor, the choice falls back to [`DEFAULT_MODEL`].
    Oldest,
}

impl Age {
    fn as_str(self) -> &'static str {
        match self {
            Age::Newest => "newest",
            Age::Oldest => "oldest",
        }
    }

    fn from_text(text: &str) -> Option<Age> {
        match text {
            "newest" => Some(Age::Newest),
            "oldest" => Some(Age::Oldest),
            _ => None,
        }
    }
}

impl fmt::Display for Age {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A model asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelChoice {
    /// The newest or oldest served model of a tier.
    Rule { tier: Tier, age: Age },
    /// An exact model name, used as given (`gemini-2.5-flash`).
    Named(String),
}

impl ModelChoice {
    /// Reads the text form a config uses: `<tier>:<age>`, with the tier
    /// `flash-lite`, `flash` or `pro` and the age `newest` or `oldest`,
    /// case ignored, spaces trimmed and a leading `gemini-` allowed
    /// (`flash-lite:oldest`, `gemini-flash:newest`). Anything else is
    /// [`ModelChoice::Named`], trimmed. Never fails.
    pub fn parse(text: &str) -> ModelChoice {
        let trimmed = text.trim();
        let lower = trimmed.to_ascii_lowercase();
        let rule = lower.strip_prefix("gemini-").unwrap_or(&lower);
        if let Some((tier, age)) = rule.split_once(':')
            && let (Some(tier), Some(age)) =
                (Tier::from_text(tier.trim()), Age::from_text(age.trim()))
        {
            return ModelChoice::Rule { tier, age };
        }
        ModelChoice::Named(trimmed.to_string())
    }
}

impl fmt::Display for ModelChoice {
    /// The text form: `flash-lite:oldest`, or the name as given.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelChoice::Rule { tier, age } => write!(f, "{tier}:{age}"),
            ModelChoice::Named(name) => f.write_str(name),
        }
    }
}

/// What a choice resolved to.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Resolved {
    /// The model name, without `models/`.
    pub model: String,
    /// How it was chosen.
    pub how: How,
}

/// How a choice was resolved.
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
    /// Nothing matched: [`DEFAULT_MODEL`].
    Fallback,
}

/// The models an account's key can use for text, as the API listed them.
#[derive(Clone, Debug, Default)]
pub struct ModelCatalog {
    /// The listed names that support `generateContent`, without `models/`.
    names: BTreeSet<String>,
}

impl ModelCatalog {
    /// A catalog of these names (each with or without `models/`), all taken
    /// as supporting `generateContent`: for tests and offline callers.
    pub fn from_names<I, S>(names: I) -> ModelCatalog
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        ModelCatalog {
            names: names
                .into_iter()
                .map(|name| bare_name(name.as_ref()).to_string())
                .filter(|name| !name.is_empty())
                .collect(),
        }
    }

    /// Lists the key's models: `GET …/v1beta/models?pageSize=1000`, every
    /// page, the key in `x-goog-api-key`, keeping those whose
    /// `supportedGenerationMethods` has `generateContent`. Errors are the
    /// HTTP or JSON failure as text, never containing the key. Call it once
    /// and keep the catalog for the life of the process (decision 7).
    pub async fn fetch(api_key: &str) -> Result<ModelCatalog, String> {
        fetch_from(MODELS_URL, api_key).await
    }

    /// The names listed, sorted.
    pub fn names(&self) -> Vec<String> {
        self.names.iter().cloned().collect()
    }

    /// The model name `choice` resolves to (without `models/`), and how
    /// (decisions 3 to 5). An empty catalog counts as no list, as
    /// [`resolve_offline`].
    ///
    /// Logs one `info` line naming the choice and the model, or one `warn`
    /// for a fallback or a named model the list lacks (decision 6).
    pub fn resolve(&self, choice: &ModelChoice) -> Resolved {
        if self.names.is_empty() {
            return resolve_offline(choice);
        }
        let resolved = self.resolve_listed(choice);
        log_resolution(choice, &resolved, true);
        resolved
    }

    fn resolve_listed(&self, choice: &ModelChoice) -> Resolved {
        match choice {
            ModelChoice::Named(name) => {
                let model = bare_name(name).to_string();
                let how = if self.names.contains(&model) {
                    How::Named
                } else {
                    How::NamedMissing
                };
                Resolved { model, how }
            }
            ModelChoice::Rule { tier, age } => self.resolve_rule(*tier, *age),
        }
    }

    fn resolve_rule(&self, tier: Tier, age: Age) -> Resolved {
        match self.pick_rule(tier, age, &[]) {
            Some((model, how)) => Resolved { model, how },
            None => fallback(),
        }
    }

    /// The model the rule picks with the `refused` names left out, and how,
    /// or `None` when the tier has nothing left: decision 4 with the floor,
    /// without the fallback and without logging. The refused-model
    /// amendment's next choice: for *oldest* the next lowest stable version
    /// at or above the floor, for *newest* the next highest after the alias.
    pub(crate) fn pick_rule(
        &self,
        tier: Tier,
        age: Age,
        refused: &[String],
    ) -> Option<(String, How)> {
        let allowed = |name: &String| !refused.iter().any(|r| r == name);
        if age == Age::Newest {
            let alias = format!("gemini-{}-latest", tier.as_str());
            if self.names.contains(&alias) && allowed(&alias) {
                return Some((alias, How::Alias));
            }
        }
        let stable = self
            .names
            .iter()
            .filter(|name| allowed(name))
            .filter_map(|name| StableName::parse(name).map(|parsed| (parsed, name)))
            .filter(|(parsed, _)| parsed.tier == tier)
            .filter(|(parsed, _)| age != Age::Oldest || parsed.at_or_above_floor());
        let picked = match age {
            Age::Newest => stable.max_by_key(|(parsed, _)| parsed.newest_key()),
            Age::Oldest => stable.min_by_key(|(parsed, _)| parsed.oldest_key()),
        };
        picked.map(|(_, name)| (name.clone(), How::Matched))
    }
}

/// Whether Google's suggested replacement `name` may stand in for the rule
/// `tier`/`age`: the same tier, stable, and for *oldest* at or above
/// [`OLDEST_FLOOR`] (the refused-model amendment, item 2).
pub(crate) fn suggestion_fits(name: &str, tier: Tier, age: Age) -> bool {
    match StableName::parse(bare_name(name)) {
        Some(parsed) => parsed.tier == tier && (age != Age::Oldest || parsed.at_or_above_floor()),
        None => false,
    }
}

/// Resolves `choice` without a catalog: a `Named` choice as given, a rule
/// to [`DEFAULT_MODEL`] (decision 5). Logs as [`ModelCatalog::resolve`].
pub fn resolve_offline(choice: &ModelChoice) -> Resolved {
    let resolved = match choice {
        ModelChoice::Named(name) => Resolved {
            model: bare_name(name).to_string(),
            how: How::Named,
        },
        ModelChoice::Rule { .. } => fallback(),
    };
    log_resolution(choice, &resolved, false);
    resolved
}

/// One line per resolution (decision 6); `listed` says whether there was
/// a model list to resolve against. Never given the key.
fn log_resolution(choice: &ModelChoice, resolved: &Resolved, listed: bool) {
    let model = &resolved.model;
    match resolved.how {
        How::Alias | How::Matched | How::Named => {
            log::info!("gemini: {choice} is {model}");
        }
        How::NamedMissing => {
            log::warn!("gemini: {choice} is not in the key's model list; using {model} as given");
        }
        How::Fallback => {
            let why = match (choice, listed) {
                (
                    ModelChoice::Rule {
                        tier,
                        age: Age::Oldest,
                    },
                    true,
                ) => {
                    let (major, minor) = OLDEST_FLOOR;
                    format!(
                        "no served {tier} model at or above {major}.{minor} in the key's model list"
                    )
                }
                (ModelChoice::Rule { tier, .. }, true) => {
                    format!("no served {tier} model in the key's model list")
                }
                _ => "no model list".to_string(),
            };
            log::warn!("gemini: {choice}: {why}; using {model}");
        }
    }
}

fn fallback() -> Resolved {
    Resolved {
        model: DEFAULT_MODEL.to_string(),
        how: How::Fallback,
    }
}

/// One page of `GET /v1beta/models`.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelsPage {
    #[serde(default)]
    models: Vec<ListedModel>,
    #[serde(default)]
    next_page_token: Option<String>,
}

/// One listed model; `displayName` and `version` are listed too, but the
/// match reads only the name.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ListedModel {
    name: String,
    #[serde(default)]
    supported_generation_methods: Vec<String>,
}

/// [`ModelCatalog::fetch`] against `base` (`…/v1beta`), so tests can serve
/// the list locally.
async fn fetch_from(base: &str, api_key: &str) -> Result<ModelCatalog, String> {
    let scrub = |text: String| {
        if api_key.is_empty() {
            text
        } else {
            text.replace(api_key, "<key>")
        }
    };
    let client = reqwest::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .build()
        .map_err(|e| scrub(format!("gemini: listing models: {e}")))?;
    let url = format!("{}/models", base.trim_end_matches('/'));
    let mut names = BTreeSet::new();
    let mut page_token: Option<String> = None;
    loop {
        let mut query = vec![("pageSize", PAGE_SIZE.to_string())];
        if let Some(token) = &page_token {
            query.push(("pageToken", token.clone()));
        }
        let page = fetch_page(&client, &url, &query, api_key)
            .compat()
            .await
            .map_err(scrub)?;
        names.extend(
            page.models
                .into_iter()
                .filter(|model| {
                    model
                        .supported_generation_methods
                        .iter()
                        .any(|method| method == "generateContent")
                })
                .map(|model| bare_name(&model.name).to_string())
                .filter(|name| !name.is_empty()),
        );
        match page.next_page_token {
            Some(token) if !token.is_empty() => page_token = Some(token),
            _ => break,
        }
    }
    Ok(ModelCatalog { names })
}

/// One page; errors may hold the key until the caller scrubs them.
async fn fetch_page(
    client: &reqwest::Client,
    url: &str,
    query: &[(&str, String)],
    api_key: &str,
) -> Result<ModelsPage, String> {
    let response = client
        .get(url)
        .query(query)
        .header("x-goog-api-key", api_key)
        .send()
        .await
        .map_err(|e| format!("gemini: listing models: {}", e.without_url()))?;
    let status = response.status();
    let body = response
        .bytes()
        .await
        .map_err(|e| format!("gemini: listing models: {}", e.without_url()))?;
    if !status.is_success() {
        let text: String = String::from_utf8_lossy(&body)
            .chars()
            .take(ERROR_BODY_CHARS)
            .collect();
        return Err(format!(
            "gemini: listing models: HTTP {status}: {}",
            text.trim()
        ));
    }
    serde_json::from_slice(&body).map_err(|e| format!("gemini: listing models: bad JSON: {e}"))
}

/// A name trimmed and without a leading `models/`.
pub(crate) fn bare_name(name: &str) -> &str {
    let name = name.trim();
    name.strip_prefix("models/").unwrap_or(name)
}

/// A stable model name read as `gemini-<version>-<tier>[-<revision>]`
/// (decision 3).
#[derive(Debug, PartialEq, Eq)]
struct StableName {
    major: u64,
    minor: u64,
    tier: Tier,
    /// A digits-only revision suffix (`-001`), if any.
    revision: Option<u64>,
}

impl StableName {
    /// `None` for names not in this shape, and for any suffix that is not
    /// a digits-only revision (`-preview…`, `-exp…`, `-tts`, …).
    fn parse(name: &str) -> Option<StableName> {
        let rest = name.strip_prefix("gemini-")?;
        let (version, rest) = rest.split_once('-')?;
        let (major, minor) = match version.split_once('.') {
            Some((major, minor)) => (digits(major)?, digits(minor)?),
            None => (digits(version)?, 0),
        };
        // The longest tier first, so `flash-lite` is never read as `flash`.
        let (tier, rest) = [Tier::FlashLite, Tier::Flash, Tier::Pro]
            .into_iter()
            .find_map(|tier| {
                let rest = rest.strip_prefix(tier.as_str())?;
                (rest.is_empty() || rest.starts_with('-')).then_some((tier, rest))
            })?;
        let revision = match rest.strip_prefix('-') {
            None => None,
            Some(suffix) => Some(digits(suffix)?),
        };
        Some(StableName {
            major,
            minor,
            tier,
            revision,
        })
    }

    /// Highest version wins; among one version, no revision beats any
    /// revision, then the highest revision.
    fn newest_key(&self) -> (u64, u64, bool, u64) {
        (
            self.major,
            self.minor,
            self.revision.is_none(),
            self.revision.unwrap_or(0),
        )
    }

    /// Whether the version is at or above [`OLDEST_FLOOR`].
    fn at_or_above_floor(&self) -> bool {
        (self.major, self.minor) >= OLDEST_FLOOR
    }

    /// Lowest version wins; among one version, no revision beats any
    /// revision, then the lowest revision.
    fn oldest_key(&self) -> (u64, u64, bool, u64) {
        (
            self.major,
            self.minor,
            self.revision.is_some(),
            self.revision.unwrap_or(0),
        )
    }
}

fn digits(text: &str) -> Option<u64> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{TestServer, captured};

    const LIST: &[&str] = &[
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-lite",
        "models/gemini-2.5-flash",
        "models/gemini-2.5-flash-lite",
        "models/gemini-2.5-pro",
        "models/gemini-3.8-flash",
        "models/gemini-3.8-flash-preview",
        "models/gemini-3.0-flash-lite",
        "models/gemini-3.5-flash-lite",
        "models/gemini-3.1-pro",
        "models/gemini-2.5-flash-001",
        "models/gemini-flash-latest",
        "models/text-embedding-004",
    ];

    fn catalog() -> ModelCatalog {
        ModelCatalog::from_names(LIST)
    }

    fn rule(tier: Tier, age: Age) -> ModelChoice {
        ModelChoice::Rule { tier, age }
    }

    fn resolved(model: &str, how: How) -> Resolved {
        Resolved {
            model: model.to_string(),
            how,
        }
    }

    #[test]
    fn each_tier_and_age_resolves_over_the_list() {
        let catalog = catalog();
        assert_eq!(
            catalog.resolve(&rule(Tier::Flash, Age::Newest)),
            resolved("gemini-flash-latest", How::Alias)
        );
        assert_eq!(
            catalog.resolve(&rule(Tier::Flash, Age::Oldest)),
            resolved("gemini-3.8-flash", How::Matched)
        );
        assert_eq!(
            catalog.resolve(&rule(Tier::FlashLite, Age::Newest)),
            resolved("gemini-3.5-flash-lite", How::Matched)
        );
        assert_eq!(
            catalog.resolve(&rule(Tier::FlashLite, Age::Oldest)),
            resolved("gemini-3.0-flash-lite", How::Matched)
        );
        assert_eq!(
            catalog.resolve(&rule(Tier::Pro, Age::Newest)),
            resolved("gemini-3.1-pro", How::Matched)
        );
        assert_eq!(
            catalog.resolve(&rule(Tier::Pro, Age::Oldest)),
            resolved("gemini-3.1-pro", How::Matched)
        );
    }

    #[test]
    fn flash_lite_is_never_read_as_flash() {
        let catalog = ModelCatalog::from_names(["gemini-3.0-flash-lite", "gemini-3.5-flash"]);
        assert_eq!(
            catalog.resolve(&rule(Tier::Flash, Age::Oldest)).model,
            "gemini-3.5-flash"
        );
        let only_lite = ModelCatalog::from_names(["gemini-2.0-flash-lite"]);
        assert_eq!(
            only_lite.resolve(&rule(Tier::Flash, Age::Newest)),
            resolved(DEFAULT_MODEL, How::Fallback)
        );
    }

    #[test]
    fn previews_and_other_suffixes_are_skipped() {
        let catalog = ModelCatalog::from_names([
            "gemini-3.5-flash",
            "gemini-3.0-flash-preview",
            "gemini-3.8-flash-preview",
            "gemini-4.0-flash-exp-0827",
            "gemini-4.0-flash-tts",
            "gemini-4.0-flash-image",
            "gemini-4.0-flash-live-001",
            "gemini-4.0-flash-native-audio-dialog",
            "gemini-4.0-flash-thinking-exp",
            "gemini-1.0-flash-preview-02-05",
        ]);
        assert_eq!(
            catalog.resolve(&rule(Tier::Flash, Age::Newest)),
            resolved("gemini-3.5-flash", How::Matched)
        );
        assert_eq!(
            catalog.resolve(&rule(Tier::Flash, Age::Oldest)),
            resolved("gemini-3.5-flash", How::Matched)
        );
        // Without the alias, newest Flash in the plan's list is the
        // stable 3.8, not the preview.
        let no_alias = ModelCatalog::from_names(
            LIST.iter()
                .filter(|name| !name.ends_with("-latest"))
                .copied(),
        );
        assert_eq!(
            no_alias.resolve(&rule(Tier::Flash, Age::Newest)),
            resolved("gemini-3.8-flash", How::Matched)
        );
    }

    #[test]
    fn a_tier_with_only_previews_falls_back() {
        let catalog = ModelCatalog::from_names(["gemini-3-pro-preview", "gemini-2.5-flash"]);
        assert_eq!(
            catalog.resolve(&rule(Tier::Pro, Age::Newest)),
            resolved(DEFAULT_MODEL, How::Fallback)
        );
    }

    #[test]
    fn the_alias_is_preferred_for_newest_only() {
        let catalog = ModelCatalog::from_names([
            "gemini-flash-lite-latest",
            "gemini-3.0-flash-lite",
            "gemini-9.9-flash-lite",
        ]);
        assert_eq!(
            catalog.resolve(&rule(Tier::FlashLite, Age::Newest)),
            resolved("gemini-flash-lite-latest", How::Alias)
        );
        assert_eq!(
            catalog.resolve(&rule(Tier::FlashLite, Age::Oldest)),
            resolved("gemini-3.0-flash-lite", How::Matched)
        );
        // Only the alias: oldest has no versioned model to pick.
        let only_alias = ModelCatalog::from_names(["gemini-pro-latest"]);
        assert_eq!(
            only_alias.resolve(&rule(Tier::Pro, Age::Oldest)),
            resolved(DEFAULT_MODEL, How::Fallback)
        );
    }

    #[test]
    fn revisions_order_within_a_version() {
        let catalog = ModelCatalog::from_names([
            "gemini-3.5-flash-002",
            "gemini-3.5-flash-001",
            "gemini-3.5-flash",
        ]);
        assert_eq!(
            catalog.resolve(&rule(Tier::Flash, Age::Newest)).model,
            "gemini-3.5-flash"
        );
        assert_eq!(
            catalog.resolve(&rule(Tier::Flash, Age::Oldest)).model,
            "gemini-3.5-flash"
        );
        let revisions_only =
            ModelCatalog::from_names(["gemini-3.5-flash-002", "gemini-3.5-flash-001"]);
        assert_eq!(
            revisions_only
                .resolve(&rule(Tier::Flash, Age::Newest))
                .model,
            "gemini-3.5-flash-002"
        );
        assert_eq!(
            revisions_only
                .resolve(&rule(Tier::Flash, Age::Oldest))
                .model,
            "gemini-3.5-flash-001"
        );
    }

    #[test]
    fn versions_compare_numerically() {
        let catalog = ModelCatalog::from_names([
            "gemini-3-flash",
            "gemini-3.10-flash",
            "gemini-3.9-flash",
            "gemini-10.0-flash",
        ]);
        assert_eq!(
            catalog.resolve(&rule(Tier::Flash, Age::Newest)).model,
            "gemini-10.0-flash"
        );
        assert_eq!(
            catalog.resolve(&rule(Tier::Flash, Age::Oldest)).model,
            "gemini-3-flash"
        );
    }

    #[test]
    fn the_floor_is_the_first_3x_model() {
        assert_eq!(OLDEST_FLOOR, (3, 0));
    }

    #[test]
    fn oldest_picks_the_lowest_version_at_or_above_the_floor() {
        let catalog = ModelCatalog::from_names([
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash-lite",
            "gemini-3.2-flash-lite",
            "gemini-3.0-flash-lite-001",
            "gemini-4.0-flash-lite",
        ]);
        assert_eq!(
            catalog.resolve(&rule(Tier::FlashLite, Age::Oldest)),
            resolved("gemini-3.0-flash-lite-001", How::Matched)
        );
    }

    #[test]
    fn oldest_with_only_versions_below_the_floor_falls_back() {
        let catalog = ModelCatalog::from_names([
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash-lite",
            "gemini-3.5-flash",
        ]);
        let lines = captured(|| {
            assert_eq!(
                catalog.resolve(&rule(Tier::FlashLite, Age::Oldest)),
                resolved(DEFAULT_MODEL, How::Fallback)
            );
        });
        assert_eq!(
            lines,
            vec![(
                log::Level::Warn,
                format!(
                    "gemini: flash-lite:oldest: no served flash-lite model at or above 3.0 \
                     in the key's model list; using {DEFAULT_MODEL}"
                )
            )]
        );
    }

    #[test]
    fn the_owners_list_gives_the_first_3x_flash_lite_for_oldest() {
        let catalog = ModelCatalog::from_names([
            "models/gemini-2.5-flash-lite",
            "models/gemini-3.5-flash-lite",
            "models/gemini-3.8-flash-lite",
            "models/gemini-flash-lite-latest",
        ]);
        assert_eq!(
            catalog.resolve(&rule(Tier::FlashLite, Age::Oldest)),
            resolved("gemini-3.5-flash-lite", How::Matched)
        );
        assert_eq!(
            catalog.resolve(&rule(Tier::FlashLite, Age::Newest)),
            resolved("gemini-flash-lite-latest", How::Alias)
        );
    }

    #[test]
    fn newest_is_unaffected_by_the_floor() {
        let catalog = ModelCatalog::from_names(["gemini-2.0-flash", "gemini-2.5-flash"]);
        assert_eq!(
            catalog.resolve(&rule(Tier::Flash, Age::Newest)),
            resolved("gemini-2.5-flash", How::Matched)
        );
        assert_eq!(
            catalog.resolve(&rule(Tier::Flash, Age::Oldest)),
            resolved(DEFAULT_MODEL, How::Fallback)
        );
    }

    #[test]
    fn a_named_model_below_the_floor_is_used_as_given() {
        let catalog = ModelCatalog::from_names(["gemini-2.5-flash-lite", "gemini-3.5-flash-lite"]);
        assert_eq!(
            catalog.resolve(&ModelChoice::Named("gemini-2.5-flash-lite".into())),
            resolved("gemini-2.5-flash-lite", How::Named)
        );
        assert_eq!(
            resolve_offline(&ModelChoice::Named("gemini-2.0-flash".into())),
            resolved("gemini-2.0-flash", How::Named)
        );
    }

    #[test]
    fn names_not_in_the_shape_never_match() {
        let catalog = ModelCatalog::from_names([
            "text-embedding-004",
            "gemma-3-27b-it",
            "learnlm-2.0-flash-experimental",
            "gemini-embedding-001",
            "gemini-flash",
        ]);
        for tier in [Tier::FlashLite, Tier::Flash, Tier::Pro] {
            for age in [Age::Newest, Age::Oldest] {
                assert_eq!(
                    catalog.resolve(&rule(tier, age)),
                    resolved(DEFAULT_MODEL, How::Fallback)
                );
            }
        }
    }

    #[test]
    fn a_named_model_present_and_missing() {
        let catalog = catalog();
        assert_eq!(
            catalog.resolve(&ModelChoice::Named("gemini-2.5-flash".into())),
            resolved("gemini-2.5-flash", How::Named)
        );
        assert_eq!(
            catalog.resolve(&ModelChoice::Named("models/gemini-2.5-pro".into())),
            resolved("gemini-2.5-pro", How::Named)
        );
        assert_eq!(
            catalog.resolve(&ModelChoice::Named("gemini-3.8-flash-lite".into())),
            resolved("gemini-3.8-flash-lite", How::NamedMissing)
        );
    }

    #[test]
    fn without_a_list_names_pass_and_rules_fall_back() {
        assert_eq!(
            resolve_offline(&ModelChoice::Named("gemini-3.8-flash-lite".into())),
            resolved("gemini-3.8-flash-lite", How::Named)
        );
        assert_eq!(
            resolve_offline(&rule(Tier::Flash, Age::Newest)),
            resolved(DEFAULT_MODEL, How::Fallback)
        );
        let empty = ModelCatalog::default();
        assert_eq!(
            empty.resolve(&ModelChoice::Named("gemini-2.5-flash".into())),
            resolved("gemini-2.5-flash", How::Named)
        );
        assert_eq!(
            empty.resolve(&rule(Tier::FlashLite, Age::Oldest)),
            resolved(DEFAULT_MODEL, How::Fallback)
        );
    }

    #[test]
    fn names_are_sorted_and_without_the_prefix() {
        let catalog = ModelCatalog::from_names(["models/gemini-2.5-pro", "gemini-2.0-flash", " "]);
        assert_eq!(catalog.names(), vec!["gemini-2.0-flash", "gemini-2.5-pro"]);
    }

    #[test]
    fn the_text_form_round_trips() {
        for tier in [Tier::FlashLite, Tier::Flash, Tier::Pro] {
            for age in [Age::Newest, Age::Oldest] {
                let choice = rule(tier, age);
                assert_eq!(ModelChoice::parse(&choice.to_string()), choice);
            }
        }
        assert_eq!(
            rule(Tier::FlashLite, Age::Oldest).to_string(),
            "flash-lite:oldest"
        );
        assert_eq!(rule(Tier::Flash, Age::Newest).to_string(), "flash:newest");
        let named = ModelChoice::Named("gemini-2.5-flash".into());
        assert_eq!(ModelChoice::parse(&named.to_string()), named);
    }

    #[test]
    fn the_text_form_ignores_case_and_spaces() {
        assert_eq!(
            ModelChoice::parse("  Flash-Lite:OLDEST "),
            rule(Tier::FlashLite, Age::Oldest)
        );
        assert_eq!(
            ModelChoice::parse("flash : newest"),
            rule(Tier::Flash, Age::Newest)
        );
        assert_eq!(
            ModelChoice::parse("Gemini-Pro:Newest"),
            rule(Tier::Pro, Age::Newest)
        );
        assert_eq!(
            ModelChoice::parse("gemini-flash:newest"),
            rule(Tier::Flash, Age::Newest)
        );
    }

    #[test]
    fn anything_else_is_named() {
        assert_eq!(
            ModelChoice::parse(" gemini-2.5-Flash "),
            ModelChoice::Named("gemini-2.5-Flash".into())
        );
        assert_eq!(
            ModelChoice::parse("flash:middle"),
            ModelChoice::Named("flash:middle".into())
        );
        assert_eq!(
            ModelChoice::parse("ultra:newest"),
            ModelChoice::Named("ultra:newest".into())
        );
        assert_eq!(ModelChoice::parse(""), ModelChoice::Named(String::new()));
    }

    const KEY: &str = "test-key-SECRET-123";

    #[tokio::test]
    async fn fetch_reads_every_page_and_keeps_generate_content_models() {
        let page1 = r#"{"models": [
            {"name": "models/gemini-2.5-flash", "displayName": "Gemini 2.5 Flash",
             "version": "001", "supportedGenerationMethods": ["generateContent", "countTokens"]},
            {"name": "models/text-embedding-004", "displayName": "Text Embedding 004",
             "version": "004", "supportedGenerationMethods": ["embedContent"]}
        ], "nextPageToken": "page 2/token"}"#;
        let page2 = r#"{"models": [
            {"name": "models/gemini-2.0-flash-lite", "displayName": "Gemini 2.0 Flash-Lite",
             "version": "2.0", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/aqa", "displayName": "AQA", "version": "001"}
        ]}"#;
        let server = TestServer::start(vec![(200, page1.into()), (200, page2.into())]);
        let catalog = fetch_from(&server.base, KEY).await.unwrap();
        assert_eq!(
            catalog.names(),
            vec!["gemini-2.0-flash-lite", "gemini-2.5-flash"]
        );
        let requests = server.requests();
        assert_eq!(requests.len(), 2);
        assert!(requests[0].starts_with("GET /v1beta/models?pageSize=1000 HTTP/1.1"));
        assert!(
            requests[1].starts_with("GET /v1beta/models?pageSize=1000&pageToken=page+2%2Ftoken ")
        );
        for request in &requests {
            assert!(
                request
                    .to_ascii_lowercase()
                    .contains(&format!("x-goog-api-key: {}", KEY.to_ascii_lowercase()))
            );
        }
    }

    #[tokio::test]
    async fn fetch_gives_an_http_error_without_the_key() {
        let body = format!(
            r#"{{"error": {{"code": 403, "message": "API key {KEY} not valid", "status": "PERMISSION_DENIED"}}}}"#
        );
        let server = TestServer::start(vec![(403, body)]);
        let error = fetch_from(&server.base, KEY).await.unwrap_err();
        server.requests();
        assert!(error.contains("HTTP 403"), "{error}");
        assert!(error.contains("PERMISSION_DENIED"), "{error}");
        assert!(!error.contains(KEY), "{error}");
    }

    #[tokio::test]
    async fn fetch_gives_bad_json_as_an_error_without_the_key() {
        let server = TestServer::start(vec![(200, format!("{{\"models\": [{KEY}"))]);
        let error = fetch_from(&server.base, KEY).await.unwrap_err();
        server.requests();
        assert!(error.contains("bad JSON"), "{error}");
        assert!(!error.contains(KEY), "{error}");
    }

    #[tokio::test]
    async fn fetch_gives_a_connection_failure_as_an_error() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}/v1beta", listener.local_addr().unwrap());
        drop(listener);
        let error = fetch_from(&base, KEY).await.unwrap_err();
        assert!(error.starts_with("gemini: listing models"), "{error}");
        assert!(!error.contains(KEY), "{error}");
    }

    #[test]
    fn a_match_logs_one_info_line() {
        let catalog = catalog();
        let lines = captured(|| {
            catalog.resolve(&rule(Tier::FlashLite, Age::Oldest));
        });
        assert_eq!(
            lines,
            vec![(
                log::Level::Info,
                "gemini: flash-lite:oldest is gemini-3.0-flash-lite".to_string()
            )]
        );
        let lines = captured(|| {
            catalog.resolve(&ModelChoice::Named("gemini-2.5-pro".into()));
        });
        assert_eq!(
            lines,
            vec![(
                log::Level::Info,
                "gemini: gemini-2.5-pro is gemini-2.5-pro".to_string()
            )]
        );
    }

    #[test]
    fn a_fallback_logs_one_warning() {
        let no_pro = ModelCatalog::from_names(["gemini-2.5-flash"]);
        let lines = captured(|| {
            no_pro.resolve(&rule(Tier::Pro, Age::Newest));
        });
        assert_eq!(
            lines,
            vec![(
                log::Level::Warn,
                format!(
                    "gemini: pro:newest: no served pro model in the key's model list; \
                     using {DEFAULT_MODEL}"
                )
            )]
        );
        let lines = captured(|| {
            resolve_offline(&rule(Tier::Flash, Age::Newest));
        });
        assert_eq!(
            lines,
            vec![(
                log::Level::Warn,
                format!("gemini: flash:newest: no model list; using {DEFAULT_MODEL}")
            )]
        );
    }

    #[test]
    fn a_missing_named_model_logs_one_warning() {
        let catalog = catalog();
        let lines = captured(|| {
            catalog.resolve(&ModelChoice::Named("gemini-3.8-flash-lite".into()));
        });
        assert_eq!(
            lines,
            vec![(
                log::Level::Warn,
                "gemini: gemini-3.8-flash-lite is not in the key's model list; \
                 using gemini-3.8-flash-lite as given"
                    .to_string()
            )]
        );
    }

    #[tokio::test]
    async fn fetching_and_resolving_never_log_the_key() {
        let page = r#"{"models": [{"name": "models/gemini-2.5-flash",
            "supportedGenerationMethods": ["generateContent"]}]}"#;
        let server = TestServer::start(vec![(200, page.into())]);
        let catalog = fetch_from(&server.base, KEY).await.unwrap();
        server.requests();
        let lines = captured(|| {
            catalog.resolve(&rule(Tier::Flash, Age::Newest));
            catalog.resolve(&rule(Tier::Pro, Age::Oldest));
            catalog.resolve(&ModelChoice::Named("gemini-x".into()));
        });
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert!(lines.iter().all(|(_, text)| !text.contains(KEY)));
    }
}
