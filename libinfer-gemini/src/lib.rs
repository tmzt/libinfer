//! The Gemini provider for `libinfer`: [`GeminiReasoningEngine`], built on
//! `gemini-rust` directly
//! (docs/design/20261022_07_LIBINFER_PROVIDERS_PLAN.md, decisions 1 and 2).
//!
//! Until tools are provider-neutral (that plan's step 4), the engine takes
//! `gemini-rust` types, which this crate re-exports.

mod choice;
#[cfg(test)]
mod testutil;

use std::sync::{Mutex, MutexGuard};

use async_compat::CompatExt;
use async_trait::async_trait;
use libinfer::{Message, ReasoningEngine, ToolExecutor};

use gemini_rust::{ClientError, ContentBuilder};

pub use gemini_rust::{
    Content, FunctionCall, FunctionCallingMode, FunctionDeclaration, Gemini,
    Message as GeminiMessage, Model, Role, Tool,
};

pub use choice::{
    Age, How, ModelCatalog, ModelChoice, OLDEST_FLOOR, Resolved, Tier, resolve_offline,
};

/// The default number of tool-calling steps allowed in one turn.
pub const DEFAULT_MAX_TURNS: usize = 5;

/// The most times an engine built from a choice replaces a refused model in
/// its life (the refused-model amendment, item 3).
const MAX_MODEL_SWITCHES: usize = 3;

/// The model used when none is configured (decision 3; CHANGES.md
/// 2026-10-23: Google no longer serves `gemini-2.5-flash` to new keys).
pub const DEFAULT_MODEL: &str = "gemini-3.8-flash";

/// The `gemini-rust` model for a configured name such as `gemini-2.5-pro`
/// (decision 3). A leading `models/` is optional, and an empty name means
/// [`DEFAULT_MODEL`]. A name `gemini-rust` does not know is passed through
/// as `Model::Custom("models/<name>")`, the resource name the API's URL
/// needs, so a new model needs no code change.
pub fn model_from_name(name: &str) -> Model {
    let name = name.trim();
    let name = name.strip_prefix("models/").unwrap_or(name);
    let name = if name.is_empty() { DEFAULT_MODEL } else { name };
    match name {
        "gemini-2.5-flash" => Model::Gemini25Flash,
        "gemini-2.5-flash-lite" => Model::Gemini25FlashLite,
        "gemini-2.5-flash-image" => Model::Gemini25FlashImage,
        "gemini-2.5-pro" => Model::Gemini25Pro,
        "gemini-3-flash-preview" => Model::Gemini3Flash,
        "gemini-3-pro-preview" => Model::Gemini3Pro,
        "gemini-3-pro-image-preview" => Model::Gemini3ProImage,
        "text-embedding-004" => Model::TextEmbedding004,
        other => Model::Custom(format!("models/{other}")),
    }
}

/// Builds a [`GeminiReasoningEngine`].
#[derive(Default)]
pub struct GeminiReasoningEngineBuilder {
    api_key: Option<String>,
    model: Option<String>,
    policy: Option<Policy>,
    /// The API's base URL (`…/v1beta/`); `None` is Google's. Set only by
    /// tests, so no test calls the real API.
    base_url: Option<reqwest::Url>,
    system_prompt: Option<String>,
    tools: Vec<Tool>,
    max_turns: Option<usize>,
}

impl GeminiReasoningEngineBuilder {
    /// The API key; with `None`, `build` reads `GEMINI_API_KEY`.
    pub fn api_key(mut self, api_key: Option<String>) -> Self {
        self.api_key = api_key;
        self
    }

    /// The model, by name (see [`model_from_name`]); unset or empty means
    /// [`DEFAULT_MODEL`]. The engine never switches away from it. Replaces
    /// an earlier [`choice`](Self::choice).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self.policy = None;
        self
    }

    /// The model, by a choice resolved against `catalog` as
    /// [`ModelCatalog::resolve`] does (and logs). The engine keeps the
    /// choice and a copy of the catalog: when the API refuses its model
    /// (HTTP 404, `NOT_FOUND`, naming the model), a rule picks again
    /// without the refused models, the next lowest at or above
    /// [`OLDEST_FLOOR`] for *oldest* or the next highest for *newest*;
    /// Google's suggested model is taken only when the rule has nothing
    /// left, and only if it is the same tier, stable and, for *oldest*, at
    /// or above the floor. The request is then sent again on the new
    /// model, at most 3 times in the engine's life, with one `warn` line
    /// per switch. A [`ModelChoice::Named`] choice never switches.
    /// Replaces an earlier [`model`](Self::model).
    pub fn choice(mut self, choice: ModelChoice, catalog: &ModelCatalog) -> Self {
        let resolved = catalog.resolve(&choice);
        self.model = Some(resolved.model);
        self.policy = Some(Policy {
            choice,
            catalog: catalog.clone(),
        });
        self
    }

    /// The system prompt sent with every request.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// The tools declared with every request.
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = tools;
        self
    }

    /// The most tool-calling steps one turn may take (default 5).
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = Some(max_turns);
        self
    }

    /// The engine. Fails, naming `GEMINI_API_KEY`, when no key was given
    /// and the variable is not set.
    pub fn build(self) -> Result<GeminiReasoningEngine, String> {
        let key = match self.api_key {
            Some(key) => key,
            None => api_key_from_env()?,
        };
        let model = model_from_name(self.model.as_deref().unwrap_or(DEFAULT_MODEL));
        let mut engine = GeminiReasoningEngine::create(
            key,
            model,
            self.base_url,
            self.system_prompt,
            self.tools,
        )?;
        engine.policy = self.policy;
        Ok(engine.with_max_turns(self.max_turns.unwrap_or(DEFAULT_MAX_TURNS)))
    }

    /// Sends requests to `base_url` (`…/v1beta/`) instead of Google's.
    #[cfg(test)]
    fn base_url(mut self, base_url: &str) -> Self {
        self.base_url = Some(reqwest::Url::parse(base_url).unwrap());
        self
    }
}

/// What an engine built from a choice keeps to replace a refused model.
struct Policy {
    choice: ModelChoice,
    catalog: ModelCatalog,
}

/// The model requests go to now, and what the engine has refused.
struct Current {
    client: Gemini,
    model: Model,
    /// The models the API refused, without `models/`.
    refused: Vec<String>,
    switches: usize,
}

/// A reasoning engine powered by Google Gemini, implementing `ReasoningEngine`.
///
/// It holds the system prompt and tools itself and builds each request from
/// them.
pub struct GeminiReasoningEngine {
    /// Kept to build a client for a replacement model; never logged.
    api_key: String,
    base_url: Option<reqwest::Url>,
    /// The model the engine was built with.
    model: Model,
    current: Mutex<Current>,
    policy: Option<Policy>,
    system_prompt: Option<String>,
    tools: Vec<Tool>,
    max_turns: usize,
}

fn api_key_from_env() -> Result<String, String> {
    std::env::var("GEMINI_API_KEY").map_err(|_| "GEMINI_API_KEY not set".to_string())
}

impl GeminiReasoningEngine {
    /// Start building an engine.
    pub fn builder() -> GeminiReasoningEngineBuilder {
        GeminiReasoningEngineBuilder::default()
    }

    /// Create a new `GeminiReasoningEngine` with the given API key and the
    /// default model.
    pub fn new(
        api_key: String,
        system_prompt: Option<String>,
        tools: Vec<Tool>,
    ) -> Result<Self, String> {
        Self::with_model(
            api_key,
            model_from_name(DEFAULT_MODEL),
            system_prompt,
            tools,
        )
    }

    /// Create a new `GeminiReasoningEngine` with the given API key and model.
    pub fn with_model(
        api_key: String,
        model: Model,
        system_prompt: Option<String>,
        tools: Vec<Tool>,
    ) -> Result<Self, String> {
        Self::create(api_key, model, None, system_prompt, tools)
    }

    fn create(
        api_key: String,
        model: Model,
        base_url: Option<reqwest::Url>,
        system_prompt: Option<String>,
        tools: Vec<Tool>,
    ) -> Result<Self, String> {
        let client = client_for(&api_key, &model, base_url.as_ref())?;
        Ok(Self {
            api_key,
            base_url,
            current: Mutex::new(Current {
                client,
                model: model.clone(),
                refused: Vec::new(),
                switches: 0,
            }),
            model,
            policy: None,
            system_prompt,
            tools,
            max_turns: DEFAULT_MAX_TURNS,
        })
    }

    /// Create from the `GEMINI_API_KEY` environment variable.
    pub fn from_env(system_prompt: Option<String>, tools: Vec<Tool>) -> Result<Self, String> {
        Self::new(api_key_from_env()?, system_prompt, tools)
    }

    /// Set the maximum multi-turn tool calling steps allowed per query.
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Override the system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// The model the engine was built with. Requests go to it unless the
    /// engine was built from a choice and the API refused it; see
    /// [`current_model`](Self::current_model).
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// The model requests go to now: the one the engine was built with, or
    /// the one that replaced it after a refusal.
    pub fn current_model(&self) -> Model {
        self.current().model.clone()
    }

    fn current(&self) -> MutexGuard<'_, Current> {
        self.current
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// A request on `client` with the system prompt and tools.
    fn request(&self, client: &Gemini) -> ContentBuilder {
        let mut builder = client.generate_content();
        if let Some(prompt) = self.system_prompt() {
            builder = builder.with_system_prompt(prompt);
        }
        for tool in self.tools() {
            builder = builder.with_tool(tool.clone());
        }
        if !self.tools().is_empty() {
            builder = builder.with_function_calling_mode(FunctionCallingMode::Auto);
        }
        builder
    }

    /// After `error` from a request sent to `sent`: the client and model to
    /// send it again on, or `None` to return the error (the refused-model
    /// amendment, items 2 to 4).
    fn replace_refused(&self, sent: &Model, error: &ClientError) -> Option<(Gemini, Model)> {
        let policy = self.policy.as_ref()?;
        let ModelChoice::Rule { tier, age } = policy.choice else {
            return None;
        };
        let sent_name = choice::bare_name(sent.as_str()).to_string();
        let suggested = refusal(error, &sent_name)?;
        let mut current = self.current();
        if current.model != *sent {
            // Another request has already replaced it.
            return Some((current.client.clone(), current.model.clone()));
        }
        if !current.refused.contains(&sent_name) {
            current.refused.push(sent_name.clone());
        }
        if current.switches >= MAX_MODEL_SWITCHES {
            return None;
        }
        let (next, how) = match policy.catalog.pick_rule(tier, age, &current.refused) {
            Some((name, _)) => (name, "policy"),
            None => match suggested {
                Some(name)
                    if choice::suggestion_fits(&name, tier, age)
                        && !current.refused.contains(&name) =>
                {
                    (name, "suggested")
                }
                _ => return None,
            },
        };
        let model = model_from_name(&next);
        let client = match client_for(&self.api_key, &model, self.base_url.as_ref()) {
            Ok(client) => client,
            Err(e) => {
                log::error!("libinfer-gemini: {e}");
                return None;
            }
        };
        current.client = client.clone();
        current.model = model.clone();
        current.switches += 1;
        log::warn!(
            "gemini: {sent_name} was refused; {} now uses {next} ({how})",
            policy.choice
        );
        Some((client, model))
    }

    /// The system prompt sent with every request, if any.
    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    /// The tools declared with every request.
    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    /// Get the maximum multi-turn tool calling steps allowed per query.
    pub fn max_turns(&self) -> usize {
        self.max_turns
    }
}

#[async_trait]
impl ReasoningEngine for GeminiReasoningEngine {
    async fn process_transcript(
        &self,
        transcript: &str,
        executor: &dyn ToolExecutor,
    ) -> Result<Option<String>, String> {
        let msg = Message {
            role: "user".to_string(),
            content: transcript.to_string(),
            name: None,
            thought_signature: None,
        };
        self.process_turns(&[msg], executor).await
    }

    async fn process_turns(
        &self,
        turns: &[Message],
        executor: &dyn ToolExecutor,
    ) -> Result<Option<String>, String> {
        if turns.is_empty() {
            return Ok(None);
        }

        let (client, mut model) = {
            let current = self.current();
            (current.client.clone(), current.model.clone())
        };
        let mut builder = self.request(&client);

        // Add all historical messages to the builder
        log::info!(
            "libinfer-gemini: Compiling {} messages into a {} request (tools: {})",
            turns.len(),
            model,
            self.tools().len()
        );
        for (i, turn) in turns.iter().enumerate() {
            log::debug!(
                "libinfer-gemini: Context msg [{}][role={}]: {}",
                i + 1,
                turn.role,
                turn.content
            );
            let role = match turn.role.as_str() {
                "model" => Role::Model,
                _ => Role::User,
            };

            let content = Content::text(&turn.content);
            builder = builder.with_message(GeminiMessage { role, content });
        }

        // Loop for tool execution within the current turn
        for turn_idx in 0..self.max_turns {
            log::info!(
                "libinfer-gemini: Dispatching Gemini API request (turn {}/{})",
                turn_idx + 1,
                self.max_turns
            );
            let response = loop {
                match builder.clone().execute().compat().await {
                    Ok(response) => break response,
                    Err(e) => {
                        if let Some((client, next)) = self.replace_refused(&model, &e) {
                            // The same request, sent again on the new model.
                            let contents = std::mem::take(&mut builder.contents);
                            builder = self.request(&client);
                            builder.contents = contents;
                            model = next;
                            continue;
                        }
                        log::error!("libinfer-gemini: Gemini API request failed: {}", e);
                        return Err(format!("gemini generate: {e}"));
                    }
                }
            };

            let calls = response.function_calls_with_thoughts();
            if let Some((fc, thought_signature)) = calls.first() {
                log::info!(
                    "libinfer-gemini: Gemini requested tool call '{}' with arguments: {}",
                    fc.name,
                    fc.args
                );
                let args_val = &fc.args;
                let exec_result = executor.execute_tool(&fc.name, args_val).await?;
                log::info!(
                    "libinfer-gemini: Tool '{}' returned output: {}",
                    fc.name,
                    exec_result
                );

                // Construct model message with thought signature if present
                let model_content = if let Some(sig) = thought_signature {
                    Content::function_call_with_thought((*fc).clone(), sig.to_string())
                } else {
                    Content::function_call((*fc).clone())
                };
                let model_msg = GeminiMessage {
                    content: model_content,
                    role: Role::Model,
                };
                builder = builder
                    .with_message(model_msg)
                    .with_function_response(&fc.name, exec_result)
                    .map_err(|e| format!("function response: {e}"))?;
            } else {
                let text = response.text().trim().to_string();
                log::info!("libinfer-gemini: Gemini final response text: '{}'", text);
                return if text.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(text))
                };
            }
        }

        Err(format!(
            "exceeded maximum multi-turn tool steps ({})",
            self.max_turns
        ))
    }
}

/// A client for `model` with `api_key`, on `base_url` when given.
fn client_for(
    api_key: &str,
    model: &Model,
    base_url: Option<&reqwest::Url>,
) -> Result<Gemini, String> {
    match base_url {
        None => Gemini::with_model(api_key, model.clone()),
        Some(base) => Gemini::with_model_and_base_url(api_key, model.clone(), base.clone()),
    }
    .map_err(|e| format!("gemini client init: {e}"))
}

/// When `error` is the API refusing `model` (HTTP 404 with status
/// `NOT_FOUND` and a message naming `models/<model>` before any
/// suggestion): `Some` of the model Google suggests instead (the
/// `models/<name>` after "use"), if any.
/// `None` for any other error.
fn refusal(error: &ClientError, model: &str) -> Option<Option<String>> {
    let ClientError::BadResponse {
        code: 404,
        description: Some(body),
    } = error
    else {
        return None;
    };
    let body: serde_json::Value = serde_json::from_str(body).ok()?;
    let error = body.get("error")?;
    if error.get("status")?.as_str()? != "NOT_FOUND" {
        return None;
    }
    let message = error.get("message")?.as_str()?;
    // The refused model is named before "use models/<suggested>".
    let (refused, suggested) = match message.split_once("use models/") {
        Some((before, after)) => (before, Some(model_name_at(after))),
        None => (message, None),
    };
    if !model_names(refused).any(|named| named == model) {
        return None;
    }
    Some(
        suggested
            .filter(|name| !name.is_empty())
            .map(str::to_string),
    )
}

/// Every name written as `models/<name>` in `text`.
fn model_names(text: &str) -> impl Iterator<Item = &str> {
    text.split("models/").skip(1).map(model_name_at)
}

/// The model name at the start of `text`: letters, digits, `-`, `.` and
/// `_`, without a trailing `.` or `-` (the end of a sentence).
fn model_name_at(text: &str) -> &str {
    let end = text
        .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_')))
        .unwrap_or(text.len());
    text[..end].trim_end_matches(['.', '-'])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Vec<Tool> {
        vec![Tool::with_functions(vec![FunctionDeclaration::new(
            "list_panes",
            "List the panes.",
            None,
        )])]
    }

    #[test]
    fn the_builder_sets_prompt_tools_and_max_turns() {
        let engine = GeminiReasoningEngine::builder()
            .api_key(Some("test_api_key".to_string()))
            .system_prompt("You are a test.")
            .tools(tools())
            .max_turns(3)
            .build()
            .unwrap();
        assert_eq!(engine.system_prompt(), Some("You are a test."));
        assert_eq!(engine.tools().len(), 1);
        assert_eq!(engine.max_turns(), 3);
    }

    #[test]
    fn the_builder_defaults() {
        let engine = GeminiReasoningEngine::builder()
            .api_key(Some("test_api_key".to_string()))
            .build()
            .unwrap();
        assert_eq!(engine.system_prompt(), None);
        assert!(engine.tools().is_empty());
        assert_eq!(engine.max_turns(), DEFAULT_MAX_TURNS);
        assert_eq!(
            engine.model(),
            &Model::Custom("models/gemini-3.8-flash".to_string())
        );
    }

    fn built_with(model: &str) -> Model {
        GeminiReasoningEngine::builder()
            .api_key(Some("test_api_key".to_string()))
            .model(model)
            .build()
            .unwrap()
            .model()
            .clone()
    }

    #[test]
    fn the_builder_passes_the_configured_model() {
        assert_eq!(built_with("gemini-2.5-pro"), Model::Gemini25Pro);
        assert_eq!(
            built_with("models/gemini-2.5-flash-lite"),
            Model::Gemini25FlashLite
        );
        assert_eq!(built_with(" gemini-3-flash-preview "), Model::Gemini3Flash);
    }

    #[test]
    fn an_empty_model_is_the_default() {
        let default = Model::Custom("models/gemini-3.8-flash".to_string());
        assert_eq!(built_with(""), default);
        assert_eq!(built_with("  "), default);
    }

    #[test]
    fn an_unknown_model_is_passed_through() {
        let model = built_with("gemini-9-ultra");
        assert_eq!(model, Model::Custom("models/gemini-9-ultra".to_string()));
        // The request URL is `<base>/<model>:generateContent`.
        assert_eq!(model.to_string(), "models/gemini-9-ultra");
        assert_eq!(
            built_with("models/gemini-9-ultra"),
            Model::Custom("models/gemini-9-ultra".to_string())
        );
    }

    // The refused-model amendment (plan 20261030_01, 15:37 EDT), item 7.

    use crate::testutil::{TestServer, captured};

    const KEY: &str = "test-key-SECRET-123";

    /// The API's refusal, as flankr's daemon logged it.
    fn refused(model: &str, suggested: &str) -> (u16, String) {
        (
            404,
            format!(
                r#"{{"error": {{"code": 404, "message": "This model models/{model} is no longer available to new users. Please update your code to use models/{suggested} instead.", "status": "NOT_FOUND"}}}}"#
            ),
        )
    }

    fn answer(text: &str) -> (u16, String) {
        (
            200,
            format!(
                r#"{{"candidates": [{{"content": {{"role": "model", "parts": [{{"text": "{text}"}}]}}}}]}}"#
            ),
        )
    }

    fn lite_oldest() -> ModelChoice {
        ModelChoice::Rule {
            tier: Tier::FlashLite,
            age: Age::Oldest,
        }
    }

    fn engine_on(
        server: &TestServer,
        choice: ModelChoice,
        names: &[&str],
    ) -> GeminiReasoningEngine {
        GeminiReasoningEngine::builder()
            .api_key(Some(KEY.to_string()))
            .base_url(&format!("{}/", server.base))
            .choice(choice, &ModelCatalog::from_names(names))
            .build()
            .unwrap()
    }

    /// The engine's answer to "hello", and this thread's log lines at
    /// `warn` and above while it ran.
    fn ask(engine: &GeminiReasoningEngine) -> (Result<Option<String>, String>, Vec<String>) {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let executor = libinfer::MockToolExecutor::new();
        let mut result = None;
        let lines = captured(|| {
            result = Some(runtime.block_on(engine.process_transcript("hello", &executor)));
        });
        for (_, text) in &lines {
            assert!(!text.contains(KEY), "{text}");
        }
        let warnings = lines
            .into_iter()
            .filter(|(level, _)| *level <= log::Level::Warn)
            .map(|(_, text)| text)
            .collect();
        (result.unwrap(), warnings)
    }

    /// The model each request went to.
    fn models_requested(server: TestServer) -> Vec<String> {
        server
            .requests()
            .iter()
            .map(|request| {
                let path = request.split(' ').nth(1).unwrap();
                let model = path.strip_prefix("/v1beta/models/").unwrap();
                model.strip_suffix(":generateContent").unwrap().to_string()
            })
            .collect()
    }

    #[test]
    fn a_refused_oldest_model_steps_to_the_next_listed_not_to_the_suggestion() {
        // The owner's case: Google suggests 3.8, but 3.6 is listed.
        let server = TestServer::start(vec![
            refused("gemini-3.5-flash-lite", "gemini-3.8-flash-lite"),
            answer("hi"),
        ]);
        let engine = engine_on(
            &server,
            lite_oldest(),
            &[
                "gemini-2.5-flash-lite",
                "gemini-3.5-flash-lite",
                "gemini-3.6-flash-lite",
                "gemini-3.8-flash-lite",
                "gemini-flash-lite-latest",
            ],
        );
        assert_eq!(
            engine.current_model().as_str(),
            "models/gemini-3.5-flash-lite"
        );
        let (result, warnings) = ask(&engine);
        assert_eq!(result, Ok(Some("hi".to_string())));
        assert_eq!(
            warnings,
            vec![
                "gemini: gemini-3.5-flash-lite was refused; flash-lite:oldest now uses \
                 gemini-3.6-flash-lite (policy)"
            ]
        );
        assert_eq!(
            models_requested(server),
            vec!["gemini-3.5-flash-lite", "gemini-3.6-flash-lite"]
        );
        assert_eq!(
            engine.current_model().as_str(),
            "models/gemini-3.6-flash-lite"
        );
        assert_eq!(engine.model().as_str(), "models/gemini-3.5-flash-lite");
    }

    #[test]
    fn the_next_request_stays_on_the_replacement() {
        let server = TestServer::start(vec![
            refused("gemini-3.5-flash-lite", "gemini-3.8-flash-lite"),
            answer("one"),
            answer("two"),
        ]);
        let engine = engine_on(
            &server,
            lite_oldest(),
            &["gemini-3.5-flash-lite", "gemini-3.6-flash-lite"],
        );
        assert_eq!(ask(&engine).0, Ok(Some("one".to_string())));
        let (result, warnings) = ask(&engine);
        assert_eq!(result, Ok(Some("two".to_string())));
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(
            models_requested(server),
            vec![
                "gemini-3.5-flash-lite",
                "gemini-3.6-flash-lite",
                "gemini-3.6-flash-lite"
            ]
        );
    }

    #[test]
    fn a_refused_newest_alias_steps_to_the_highest_version() {
        let server = TestServer::start(vec![
            refused("gemini-flash-latest", "gemini-3.5-flash"),
            answer("hi"),
        ]);
        let engine = engine_on(
            &server,
            ModelChoice::Rule {
                tier: Tier::Flash,
                age: Age::Newest,
            },
            &[
                "gemini-flash-latest",
                "gemini-3.5-flash",
                "gemini-3.8-flash",
            ],
        );
        let (result, warnings) = ask(&engine);
        assert_eq!(result, Ok(Some("hi".to_string())));
        assert_eq!(
            warnings,
            vec![
                "gemini: gemini-flash-latest was refused; flash:newest now uses \
                 gemini-3.8-flash (policy)"
            ]
        );
        assert_eq!(
            models_requested(server),
            vec!["gemini-flash-latest", "gemini-3.8-flash"]
        );
    }

    #[test]
    fn the_suggestion_is_used_when_the_rule_has_nothing_left() {
        let server = TestServer::start(vec![
            refused("gemini-3.5-flash-lite", "gemini-3.8-flash-lite"),
            answer("hi"),
        ]);
        let engine = engine_on(
            &server,
            lite_oldest(),
            &["gemini-2.5-flash-lite", "gemini-3.5-flash-lite"],
        );
        let (result, warnings) = ask(&engine);
        assert_eq!(result, Ok(Some("hi".to_string())));
        assert_eq!(
            warnings,
            vec![
                "gemini: gemini-3.5-flash-lite was refused; flash-lite:oldest now uses \
                 gemini-3.8-flash-lite (suggested)"
            ]
        );
        assert_eq!(
            models_requested(server),
            vec!["gemini-3.5-flash-lite", "gemini-3.8-flash-lite"]
        );
    }

    /// One refusal suggesting `suggested`, with nothing left in the list:
    /// the 404 is returned and nothing switches.
    fn suggestion_is_not_taken(choice: ModelChoice, model: &str, suggested: &str) {
        let server = TestServer::start(vec![refused(model, suggested)]);
        let engine = engine_on(&server, choice, &[model]);
        let (result, warnings) = ask(&engine);
        let error = result.unwrap_err();
        assert!(error.contains("code 404"), "{error}");
        assert!(error.contains("NOT_FOUND"), "{error}");
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("request failed"), "{warnings:?}");
        assert_eq!(models_requested(server), vec![model]);
        assert_eq!(engine.current_model(), model_from_name(model));
    }

    #[test]
    fn the_suggestion_is_refused_below_the_floor_or_in_another_tier() {
        suggestion_is_not_taken(
            lite_oldest(),
            "gemini-3.5-flash-lite",
            "gemini-2.5-flash-lite",
        );
        suggestion_is_not_taken(lite_oldest(), "gemini-3.5-flash-lite", "gemini-3.8-flash");
        suggestion_is_not_taken(
            lite_oldest(),
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash-lite-preview",
        );
        // The same model suggested back is not taken either.
        suggestion_is_not_taken(
            lite_oldest(),
            "gemini-3.5-flash-lite",
            "gemini-3.5-flash-lite",
        );
    }

    #[test]
    fn newest_takes_a_suggestion_below_the_floor() {
        let server = TestServer::start(vec![
            refused("gemini-3.8-flash", "gemini-2.5-flash"),
            answer("hi"),
        ]);
        let engine = engine_on(
            &server,
            ModelChoice::Rule {
                tier: Tier::Flash,
                age: Age::Newest,
            },
            &["gemini-3.8-flash"],
        );
        assert_eq!(ask(&engine).0, Ok(Some("hi".to_string())));
        assert_eq!(
            models_requested(server),
            vec!["gemini-3.8-flash", "gemini-2.5-flash"]
        );
    }

    #[test]
    fn a_named_model_is_not_switched() {
        suggestion_is_not_taken(
            ModelChoice::Named("gemini-3.5-flash-lite".into()),
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash-lite",
        );
        let server = TestServer::start(vec![refused(
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash-lite",
        )]);
        let engine = engine_on(
            &server,
            ModelChoice::Named("gemini-3.5-flash-lite".into()),
            &["gemini-3.5-flash-lite", "gemini-3.6-flash-lite"],
        );
        assert!(ask(&engine).0.is_err());
        assert_eq!(models_requested(server), vec!["gemini-3.5-flash-lite"]);
    }

    #[test]
    fn an_engine_built_by_name_is_not_switched() {
        let server = TestServer::start(vec![refused(
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash-lite",
        )]);
        let engine = GeminiReasoningEngine::builder()
            .api_key(Some(KEY.to_string()))
            .base_url(&format!("{}/", server.base))
            .choice(
                lite_oldest(),
                &ModelCatalog::from_names(["gemini-3.0-flash-lite"]),
            )
            .model("gemini-3.5-flash-lite")
            .build()
            .unwrap();
        assert!(ask(&engine).0.is_err());
        assert_eq!(models_requested(server), vec!["gemini-3.5-flash-lite"]);
    }

    #[test]
    fn at_most_three_switches() {
        let names = [
            "gemini-3.0-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3.2-flash-lite",
            "gemini-3.3-flash-lite",
            "gemini-3.4-flash-lite",
        ];
        let server = TestServer::start(
            names[..4]
                .iter()
                .map(|name| refused(name, "gemini-3.8-flash-lite"))
                .collect(),
        );
        let engine = engine_on(&server, lite_oldest(), &names);
        let (result, warnings) = ask(&engine);
        assert!(result.unwrap_err().contains("code 404"));
        let switches: Vec<_> = warnings
            .iter()
            .filter(|w| w.contains("was refused"))
            .collect();
        assert_eq!(switches.len(), 3, "{warnings:?}");
        assert_eq!(models_requested(server), names[..4].to_vec());
        assert_eq!(
            engine.current_model(),
            model_from_name("gemini-3.3-flash-lite")
        );
    }

    #[test]
    fn other_errors_do_not_switch() {
        let not_this_model = refused("gemini-3.0-flash-lite", "gemini-3.8-flash-lite");
        let other_status = (
            404,
            r#"{"error": {"code": 404, "message": "models/gemini-3.5-flash-lite: no", "status": "UNAVAILABLE"}}"#
                .to_string(),
        );
        let server_error = (
            500,
            r#"{"error": {"code": 500, "message": "models/gemini-3.5-flash-lite failed", "status": "NOT_FOUND"}}"#
                .to_string(),
        );
        for response in [not_this_model, other_status, server_error] {
            let server = TestServer::start(vec![response]);
            let engine = engine_on(
                &server,
                lite_oldest(),
                &["gemini-3.5-flash-lite", "gemini-3.6-flash-lite"],
            );
            assert!(ask(&engine).0.is_err());
            assert_eq!(models_requested(server), vec!["gemini-3.5-flash-lite"]);
        }
    }

    #[test]
    fn a_refusal_is_read_from_the_apis_error() {
        let bad = |code, body: &str| ClientError::BadResponse {
            code,
            description: Some(body.to_string()),
        };
        // The text flankr's daemon logged.
        let logged = bad(
            404,
            r#"{"error": {"code": 404, "message": "This model models/gemini-2.5-flash-lite is no longer available to new users. Please update your code to use models/gemini-3.5-flash-lite …", "status": "NOT_FOUND"}}"#,
        );
        assert_eq!(
            refusal(&logged, "gemini-2.5-flash-lite"),
            Some(Some("gemini-3.5-flash-lite".to_string()))
        );
        assert_eq!(refusal(&logged, "gemini-2.5-flash"), None);
        assert_eq!(refusal(&logged, "gemini-3.5-flash-lite"), None);
        let not_found = bad(
            404,
            r#"{"error": {"code": 404, "message": "models/gemini-3.8-flash-lite is not found for API version v1beta, or is not supported for generateContent.", "status": "NOT_FOUND"}}"#,
        );
        assert_eq!(refusal(&not_found, "gemini-3.8-flash-lite"), Some(None));
        assert_eq!(
            refusal(&bad(404, "not json"), "gemini-3.8-flash-lite"),
            None
        );
        let no_body = ClientError::BadResponse {
            code: 404,
            description: None,
        };
        assert_eq!(refusal(&no_body, "gemini-3.8-flash-lite"), None);
    }
}
