//! The Gemini provider for `libinfer`: [`GeminiReasoningEngine`], built on
//! `gemini-rust` directly
//! (docs/design/20261022_07_LIBINFER_PROVIDERS_PLAN.md, decisions 1 and 2).
//!
//! Until tools are provider-neutral (that plan's step 4), the engine takes
//! `gemini-rust` types, which this crate re-exports.

use async_compat::CompatExt;
use async_trait::async_trait;
use libinfer::{Message, ReasoningEngine, ToolExecutor};

pub use gemini_rust::{
    Content, FunctionCall, FunctionCallingMode, FunctionDeclaration, Gemini,
    Message as GeminiMessage, Model, Role, Tool,
};

/// The default number of tool-calling steps allowed in one turn.
pub const DEFAULT_MAX_TURNS: usize = 5;

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
    /// [`DEFAULT_MODEL`].
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
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
        let engine = GeminiReasoningEngine::with_model(key, model, self.system_prompt, self.tools)?;
        Ok(engine.with_max_turns(self.max_turns.unwrap_or(DEFAULT_MAX_TURNS)))
    }
}

/// A reasoning engine powered by Google Gemini, implementing `ReasoningEngine`.
///
/// It holds the system prompt and tools itself and builds each request from
/// them.
pub struct GeminiReasoningEngine {
    client: Gemini,
    model: Model,
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
        let client = Gemini::with_model(api_key, model.clone())
            .map_err(|e| format!("gemini client init: {e}"))?;
        Ok(Self {
            client,
            model,
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

    /// The model every request goes to.
    pub fn model(&self) -> &Model {
        &self.model
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

        let mut builder = self.client.generate_content();

        if let Some(prompt) = self.system_prompt() {
            builder = builder.with_system_prompt(prompt);
        }

        for tool in self.tools() {
            builder = builder.with_tool(tool.clone());
        }

        if !self.tools().is_empty() {
            builder = builder.with_function_calling_mode(FunctionCallingMode::Auto);
        }

        // Add all historical messages to the builder
        log::info!(
            "libinfer-gemini: Compiling {} messages into a {} request (tools: {})",
            turns.len(),
            self.model,
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
            let response = builder.clone().execute().compat().await.map_err(|e| {
                log::error!("libinfer-gemini: Gemini API request failed: {}", e);
                format!("gemini generate: {e}")
            })?;

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
}
