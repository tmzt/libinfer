//! A Gemini model asked for by its tier and age, or by an exact name, and
//! resolved against the models an account's key can use
//! (docs/design/20261030_01_GEMINI_MODEL_CHOICE_PLAN.md).
//!
//! The Gemini API lists no prices, so *newest* and *oldest* stand for
//! "the best, at a higher cost" and "the cheapest" (decision 8).

use std::collections::BTreeSet;
use std::fmt;

use crate::DEFAULT_MODEL;

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
    /// The oldest version still served: the cheapest.
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

    /// The names listed, sorted.
    pub fn names(&self) -> Vec<String> {
        self.names.iter().cloned().collect()
    }

    /// The model name `choice` resolves to (without `models/`), and how
    /// (decisions 3 to 5). An empty catalog counts as no list, as
    /// [`resolve_offline`].
    pub fn resolve(&self, choice: &ModelChoice) -> Resolved {
        if self.names.is_empty() {
            return resolve_offline(choice);
        }
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
        if age == Age::Newest {
            let alias = format!("gemini-{}-latest", tier.as_str());
            if self.names.contains(&alias) {
                return Resolved {
                    model: alias,
                    how: How::Alias,
                };
            }
        }
        let stable = self
            .names
            .iter()
            .filter_map(|name| StableName::parse(name).map(|parsed| (parsed, name)))
            .filter(|(parsed, _)| parsed.tier == tier);
        let picked = match age {
            Age::Newest => stable.max_by_key(|(parsed, _)| parsed.newest_key()),
            Age::Oldest => stable.min_by_key(|(parsed, _)| parsed.oldest_key()),
        };
        match picked {
            Some((_, name)) => Resolved {
                model: name.clone(),
                how: How::Matched,
            },
            None => fallback(),
        }
    }
}

/// Resolves `choice` without a catalog: a `Named` choice as given, a rule
/// to [`DEFAULT_MODEL`] (decision 5).
pub fn resolve_offline(choice: &ModelChoice) -> Resolved {
    match choice {
        ModelChoice::Named(name) => Resolved {
            model: bare_name(name).to_string(),
            how: How::Named,
        },
        ModelChoice::Rule { .. } => fallback(),
    }
}

fn fallback() -> Resolved {
    Resolved {
        model: DEFAULT_MODEL.to_string(),
        how: How::Fallback,
    }
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

    const LIST: &[&str] = &[
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-lite",
        "models/gemini-2.5-flash",
        "models/gemini-2.5-flash-lite",
        "models/gemini-2.5-pro",
        "models/gemini-3.8-flash",
        "models/gemini-3.8-flash-preview",
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
            resolved("gemini-2.0-flash", How::Matched)
        );
        assert_eq!(
            catalog.resolve(&rule(Tier::FlashLite, Age::Newest)),
            resolved("gemini-2.5-flash-lite", How::Matched)
        );
        assert_eq!(
            catalog.resolve(&rule(Tier::FlashLite, Age::Oldest)),
            resolved("gemini-2.0-flash-lite", How::Matched)
        );
        assert_eq!(
            catalog.resolve(&rule(Tier::Pro, Age::Newest)),
            resolved("gemini-2.5-pro", How::Matched)
        );
        assert_eq!(
            catalog.resolve(&rule(Tier::Pro, Age::Oldest)),
            resolved("gemini-2.5-pro", How::Matched)
        );
    }

    #[test]
    fn flash_lite_is_never_read_as_flash() {
        let catalog = ModelCatalog::from_names(["gemini-1.5-flash-lite", "gemini-2.0-flash"]);
        assert_eq!(
            catalog.resolve(&rule(Tier::Flash, Age::Oldest)).model,
            "gemini-2.0-flash"
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
            "gemini-2.5-flash",
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
            resolved("gemini-2.5-flash", How::Matched)
        );
        assert_eq!(
            catalog.resolve(&rule(Tier::Flash, Age::Oldest)),
            resolved("gemini-2.5-flash", How::Matched)
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
            "gemini-2.0-flash-lite",
            "gemini-9.9-flash-lite",
        ]);
        assert_eq!(
            catalog.resolve(&rule(Tier::FlashLite, Age::Newest)),
            resolved("gemini-flash-lite-latest", How::Alias)
        );
        assert_eq!(
            catalog.resolve(&rule(Tier::FlashLite, Age::Oldest)),
            resolved("gemini-2.0-flash-lite", How::Matched)
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
            "gemini-2.5-flash-002",
            "gemini-2.5-flash-001",
            "gemini-2.5-flash",
        ]);
        assert_eq!(
            catalog.resolve(&rule(Tier::Flash, Age::Newest)).model,
            "gemini-2.5-flash"
        );
        assert_eq!(
            catalog.resolve(&rule(Tier::Flash, Age::Oldest)).model,
            "gemini-2.5-flash"
        );
        let revisions_only =
            ModelCatalog::from_names(["gemini-2.5-flash-002", "gemini-2.5-flash-001"]);
        assert_eq!(
            revisions_only
                .resolve(&rule(Tier::Flash, Age::Newest))
                .model,
            "gemini-2.5-flash-002"
        );
        assert_eq!(
            revisions_only
                .resolve(&rule(Tier::Flash, Age::Oldest))
                .model,
            "gemini-2.5-flash-001"
        );
    }

    #[test]
    fn versions_compare_numerically() {
        let catalog = ModelCatalog::from_names([
            "gemini-2-flash",
            "gemini-2.10-flash",
            "gemini-2.9-flash",
            "gemini-10.0-flash",
        ]);
        assert_eq!(
            catalog.resolve(&rule(Tier::Flash, Age::Newest)).model,
            "gemini-10.0-flash"
        );
        assert_eq!(
            catalog.resolve(&rule(Tier::Flash, Age::Oldest)).model,
            "gemini-2-flash"
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
}
