//! A provider-neutral reasoning layer: the `ReasoningEngine`, `ToolExecutor`
//! and `TurnContributor` traits, the JSONL session store, and mock engines.
//! Providers, such as `libinfer-gemini`, are their own crates.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

#[async_trait]
pub trait ToolExecutor: Send + Sync {
    async fn execute_tool(
        &self,
        name: &str,
        args: &serde_json::Value,
    ) -> Result<serde_json::Value, String>;
}

#[async_trait]
impl<T: ToolExecutor + ?Sized> ToolExecutor for std::sync::Arc<T> {
    async fn execute_tool(
        &self,
        name: &str,
        args: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        (**self).execute_tool(name, args).await
    }
}

#[async_trait]
impl<T: ToolExecutor + ?Sized> ToolExecutor for Box<T> {
    async fn execute_tool(
        &self,
        name: &str,
        args: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        (**self).execute_tool(name, args).await
    }
}

#[async_trait]
pub trait ReasoningEngine: Send + Sync {
    async fn process_transcript(
        &self,
        transcript: &str,
        executor: &dyn ToolExecutor,
    ) -> Result<Option<String>, String>;

    async fn process_turns(
        &self,
        turns: &[Message],
        executor: &dyn ToolExecutor,
    ) -> Result<Option<String>, String>;
}

#[async_trait]
impl<T: ReasoningEngine + ?Sized> ReasoningEngine for std::sync::Arc<T> {
    async fn process_transcript(
        &self,
        transcript: &str,
        executor: &dyn ToolExecutor,
    ) -> Result<Option<String>, String> {
        (**self).process_transcript(transcript, executor).await
    }

    async fn process_turns(
        &self,
        turns: &[Message],
        executor: &dyn ToolExecutor,
    ) -> Result<Option<String>, String> {
        (**self).process_turns(turns, executor).await
    }
}

#[async_trait]
impl<T: ReasoningEngine + ?Sized> ReasoningEngine for Box<T> {
    async fn process_transcript(
        &self,
        transcript: &str,
        executor: &dyn ToolExecutor,
    ) -> Result<Option<String>, String> {
        (**self).process_transcript(transcript, executor).await
    }

    async fn process_turns(
        &self,
        turns: &[Message],
        executor: &dyn ToolExecutor,
    ) -> Result<Option<String>, String> {
        (**self).process_turns(turns, executor).await
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Message {
    pub role: String, // "user", "model", "system", "tool"
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
}

#[derive(Clone)]
pub struct SessionManager {
    cache_dir: PathBuf,
}

impl SessionManager {
    pub fn new() -> Result<Self, String> {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/Users/tmeade".to_string());
        let cache_dir = PathBuf::from(home).join(".cache").join("libinfer");
        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| format!("failed to create session cache dir: {e}"))?;
        Ok(Self { cache_dir })
    }

    pub fn with_dir(cache_dir: PathBuf) -> Result<Self, String> {
        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| format!("failed to create session cache dir: {e}"))?;
        Ok(Self { cache_dir })
    }

    fn session_path(&self, session_id: &str) -> PathBuf {
        self.cache_dir.join(format!("{}.jsonl", session_id))
    }

    pub fn load_session(&self, session_id: &str) -> Result<Vec<Message>, String> {
        let path = self.session_path(session_id);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let file = File::open(&path).map_err(|e| format!("failed to open session file: {e}"))?;
        let reader = BufReader::new(file);
        let mut messages = Vec::new();
        for (i, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| format!("failed to read line {}: {e}", i + 1))?;
            let msg: Message = serde_json::from_str(&line)
                .map_err(|e| format!("failed to deserialize message at line {}: {e}", i + 1))?;
            messages.push(msg);
        }
        Ok(messages)
    }

    pub fn append_message(&self, session_id: &str, message: &Message) -> Result<(), String> {
        let path = self.session_path(session_id);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("failed to open session file for append: {e}"))?;
        let line = serde_json::to_string(message)
            .map_err(|e| format!("failed to serialize message: {e}"))?;
        writeln!(file, "{}", line)
            .map_err(|e| format!("failed to write message to session file: {e}"))?;
        Ok(())
    }

    pub fn save_session(&self, session_id: &str, messages: &[Message]) -> Result<(), String> {
        let path = self.session_path(session_id);
        let mut file =
            File::create(&path).map_err(|e| format!("failed to create session file: {e}"))?;
        for msg in messages {
            let line = serde_json::to_string(msg)
                .map_err(|e| format!("failed to serialize message: {e}"))?;
            writeln!(file, "{}", line).map_err(|e| format!("failed to write message: {e}"))?;
        }
        Ok(())
    }
}

/// A contributor that dynamically injects context or modifies turns in a session.
#[async_trait]
pub trait TurnContributor: Send + Sync {
    /// Modify or inject turns into the conversation history.
    async fn contribute(&self, messages: &mut Vec<Message>) -> Result<(), String>;
}

/// Builder pattern for compiling session turns using contributors.
pub struct SessionBuilder {
    session_id: String,
    manager: SessionManager,
    contributors: Vec<Box<dyn TurnContributor>>,
}

impl SessionBuilder {
    pub fn new(session_id: impl Into<String>, manager: SessionManager) -> Self {
        Self {
            session_id: session_id.into(),
            manager,
            contributors: Vec::new(),
        }
    }

    /// Add a contributor to the session builder.
    pub fn with_contributor(mut self, contributor: Box<dyn TurnContributor>) -> Self {
        self.contributors.push(contributor);
        self
    }

    /// Build the final list of messages, running all turn contributors in sequence.
    pub async fn build(self) -> Result<Vec<Message>, String> {
        let mut messages = self.manager.load_session(&self.session_id)?;
        for contributor in &self.contributors {
            contributor.contribute(&mut messages).await?;
        }
        Ok(messages)
    }
}

/// A fixed reply to a set of whole utterances (plan 20261022_07, decision 4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Route {
    /// The utterances that take this route, compared after
    /// [`normalize_utterance`].
    pub phrases: Vec<String>,
    /// What the turn answers.
    pub reply: String,
}

impl Route {
    pub fn new(phrases: &[&str], reply: impl Into<String>) -> Self {
        Self {
            phrases: phrases.iter().map(|p| p.to_string()).collect(),
            reply: reply.into(),
        }
    }
}

/// `text` lowercased and trimmed, with runs of whitespace folded to one
/// space and `.`, `,`, `!` and `?` dropped.
pub fn normalize_utterance(text: &str) -> String {
    let kept: String = text
        .chars()
        .filter(|c| !matches!(c, '.' | ',' | '!' | '?'))
        .collect();
    kept.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Wraps an engine, answering some whole utterances itself
/// (plan 20260831_02, B.3; plan 20261022_07, decision 4).
///
/// A turn whose latest user message, normalised, equals one of a route's
/// phrases is answered with that route's reply: the inner engine is not
/// called and no tool runs, so a route can never send anything to an
/// agent. Every other turn goes to the inner engine unchanged.
///
/// Like every engine, it returns the reply and leaves the session to the
/// caller, which appends it as a model message the same way as any other
/// reply.
pub struct RoutingReasoningEngine {
    inner: std::sync::Arc<dyn ReasoningEngine>,
    /// Normalised phrase to reply.
    routes: std::collections::HashMap<String, String>,
}

impl RoutingReasoningEngine {
    /// Routes `routes` around `inner`. Where two routes share a phrase, the
    /// first wins.
    pub fn new(inner: std::sync::Arc<dyn ReasoningEngine>, routes: Vec<Route>) -> Self {
        let mut table = std::collections::HashMap::new();
        for route in routes {
            for phrase in &route.phrases {
                let phrase = normalize_utterance(phrase);
                if !phrase.is_empty() {
                    table.entry(phrase).or_insert_with(|| route.reply.clone());
                }
            }
        }
        Self {
            inner,
            routes: table,
        }
    }

    /// The reply for `utterance`, if it takes a route.
    pub fn route(&self, utterance: &str) -> Option<&str> {
        self.routes
            .get(&normalize_utterance(utterance))
            .map(String::as_str)
    }
}

#[async_trait]
impl ReasoningEngine for RoutingReasoningEngine {
    async fn process_transcript(
        &self,
        transcript: &str,
        executor: &dyn ToolExecutor,
    ) -> Result<Option<String>, String> {
        if let Some(reply) = self.route(transcript) {
            log::info!("libinfer: routed {transcript:?} without the engine");
            return Ok(Some(reply.to_string()));
        }
        self.inner.process_transcript(transcript, executor).await
    }

    async fn process_turns(
        &self,
        turns: &[Message],
        executor: &dyn ToolExecutor,
    ) -> Result<Option<String>, String> {
        if let Some(latest) = turns.iter().rev().find(|m| m.role == "user")
            && let Some(reply) = self.route(&latest.content)
        {
            log::info!("libinfer: routed {:?} without the engine", latest.content);
            return Ok(Some(reply.to_string()));
        }
        self.inner.process_turns(turns, executor).await
    }
}

/// Handler trait for custom mock logic.
#[async_trait]
pub trait MockHandler: Send + Sync {
    async fn handle(
        &self,
        transcript: &str,
        executor: &dyn ToolExecutor,
    ) -> Result<Option<String>, String>;
}

/// A mock reasoning engine for unit tests and local simulations without LLM network calls.
pub struct MockReasoningEngine {
    transcripts: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    executed_tools: std::sync::Arc<std::sync::Mutex<Vec<(String, serde_json::Value)>>>,
    responses: std::sync::Arc<std::sync::Mutex<Vec<Option<String>>>>,
    default_response: Option<String>,
    handler: Option<std::sync::Arc<dyn MockHandler>>,
}

impl MockReasoningEngine {
    /// Create a new mock reasoning engine with an optional canned response.
    pub fn new(default_response: Option<String>) -> Self {
        Self {
            transcripts: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            executed_tools: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            responses: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            default_response,
            handler: None,
        }
    }

    /// Create with a deterministic string response.
    pub fn with_response(response: impl Into<String>) -> Self {
        Self::new(Some(response.into()))
    }

    /// Create a mock engine that fails with the given error message.
    pub fn with_error(error: impl Into<String>) -> Self {
        struct ErrorHandler(String);
        #[async_trait]
        impl MockHandler for ErrorHandler {
            async fn handle(
                &self,
                _transcript: &str,
                _executor: &dyn ToolExecutor,
            ) -> Result<Option<String>, String> {
                Err(self.0.clone())
            }
        }

        Self {
            transcripts: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            executed_tools: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            responses: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            default_response: None,
            handler: Some(std::sync::Arc::new(ErrorHandler(error.into()))),
        }
    }

    /// Create with a custom dynamic asynchronous handler implementing `MockHandler`.
    pub fn with_handler(handler: impl MockHandler + 'static) -> Self {
        Self {
            transcripts: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            executed_tools: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            responses: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            default_response: None,
            handler: Some(std::sync::Arc::new(handler)),
        }
    }

    /// Queue multiple canned responses to be returned in sequential order.
    pub fn with_queued_responses(responses: Vec<Option<String>>) -> Self {
        Self {
            transcripts: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            executed_tools: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            responses: std::sync::Arc::new(std::sync::Mutex::new(responses)),
            default_response: None,
            handler: None,
        }
    }

    /// Retrieve all transcripts processed by this mock engine.
    pub fn recorded_transcripts(&self) -> Vec<String> {
        self.transcripts.lock().unwrap().clone()
    }

    /// Retrieve all tool calls executed through this mock engine.
    pub fn recorded_tool_calls(&self) -> Vec<(String, serde_json::Value)> {
        self.executed_tools.lock().unwrap().clone()
    }

    /// Clear recorded transcripts and tool calls.
    pub fn clear_records(&self) {
        self.transcripts.lock().unwrap().clear();
        self.executed_tools.lock().unwrap().clear();
    }
}

#[async_trait]
impl ReasoningEngine for MockReasoningEngine {
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
        let last_content = turns.last().map(|m| m.content.as_str()).unwrap_or("");

        self.transcripts
            .lock()
            .unwrap()
            .push(last_content.to_string());

        if let Some(ref handler) = self.handler {
            return handler.handle(last_content, executor).await;
        }

        let mut queue = self.responses.lock().unwrap();
        if !queue.is_empty() {
            return Ok(queue.remove(0));
        }

        if last_content.contains("force_error") {
            return Err("simulated reasoning error".to_string());
        }

        Ok(self.default_response.clone())
    }
}

/// In-memory mock tool executor for testing reasoning engines without a live Herdr socket.
#[derive(Default, Clone)]
pub struct MockToolExecutor {
    calls: std::sync::Arc<std::sync::Mutex<Vec<(String, serde_json::Value)>>>,
    canned_responses:
        std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, serde_json::Value>>>,
}

impl MockToolExecutor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_canned_response(
        self,
        tool_name: impl Into<String>,
        response: serde_json::Value,
    ) -> Self {
        self.canned_responses
            .lock()
            .unwrap()
            .insert(tool_name.into(), response);
        self
    }

    pub fn recorded_calls(&self) -> Vec<(String, serde_json::Value)> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl ToolExecutor for MockToolExecutor {
    async fn execute_tool(
        &self,
        name: &str,
        args: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        self.calls
            .lock()
            .unwrap()
            .push((name.to_string(), args.clone()));

        if let Some(resp) = self.canned_responses.lock().unwrap().get(name) {
            Ok(resp.clone())
        } else {
            Ok(serde_json::json!({
                "status": "success",
                "tool": name,
                "args": args
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_session_manager_save_load() {
        let dir = tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf()).unwrap();
        let session_id = "test_session";

        let messages = vec![
            Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
                name: None,
                thought_signature: None,
            },
            Message {
                role: "model".to_string(),
                content: "Hi there!".to_string(),
                name: None,
                thought_signature: Some("opaque_base64_sig".to_string()),
            },
        ];

        manager.save_session(session_id, &messages).unwrap();
        let loaded = manager.load_session(session_id).unwrap();
        assert_eq!(loaded, messages);

        // Test append
        let new_msg = Message {
            role: "user".to_string(),
            content: "How are you?".to_string(),
            name: None,
            thought_signature: None,
        };
        manager.append_message(session_id, &new_msg).unwrap();

        let loaded_after_append = manager.load_session(session_id).unwrap();
        assert_eq!(loaded_after_append.len(), 3);
        assert_eq!(loaded_after_append[2], new_msg);
    }

    #[tokio::test]
    async fn test_session_builder_with_contributor() {
        struct MockContributor;
        #[async_trait]
        impl TurnContributor for MockContributor {
            async fn contribute(&self, messages: &mut Vec<Message>) -> Result<(), String> {
                messages.push(Message {
                    role: "system".to_string(),
                    content: "Injected context".to_string(),
                    name: None,
                    thought_signature: None,
                });
                Ok(())
            }
        }

        let dir = tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf()).unwrap();
        let session_id = "test_builder";

        let initial_msgs = vec![Message {
            role: "user".to_string(),
            content: "Hello".to_string(),
            name: None,
            thought_signature: None,
        }];
        manager.save_session(session_id, &initial_msgs).unwrap();

        let builder =
            SessionBuilder::new(session_id, manager).with_contributor(Box::new(MockContributor));

        let built = builder.build().await.unwrap();
        assert_eq!(built.len(), 2);
        assert_eq!(built[0], initial_msgs[0]);
        assert_eq!(built[1].role, "system");
        assert_eq!(built[1].content, "Injected context");
    }

    fn user(text: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: text.to_string(),
            name: None,
            thought_signature: None,
        }
    }

    fn router() -> (std::sync::Arc<MockReasoningEngine>, RoutingReasoningEngine) {
        let inner = std::sync::Arc::new(MockReasoningEngine::with_response("From the engine."));
        let routes = vec![Route::new(
            &["never mind", "cancel", "forget it", "nothing"],
            "Okay.",
        )];
        let engine = RoutingReasoningEngine::new(inner.clone(), routes);
        (inner, engine)
    }

    #[test]
    fn normalizing_folds_case_space_and_punctuation() {
        assert_eq!(normalize_utterance("  Never   Mind. "), "never mind");
        assert_eq!(normalize_utterance("Forget it!?"), "forget it");
        assert_eq!(normalize_utterance("\tNOTHING,\n"), "nothing");
        assert_eq!(normalize_utterance("never mind ."), "never mind");
    }

    #[tokio::test]
    async fn every_phrase_is_routed_in_several_spellings() {
        let (inner, engine) = router();
        let executor = MockToolExecutor::new();
        for said in [
            "never mind",
            "Never mind.",
            "NEVER  MIND!",
            "cancel",
            "  cancel ",
            "Cancel?",
            "forget it",
            "Forget it.",
            "FORGET, IT",
            "nothing",
            "NOTHING!",
            "Nothing.",
        ] {
            let reply = engine.process_turns(&[user(said)], &executor).await;
            assert_eq!(reply, Ok(Some("Okay.".to_string())), "{said:?}");
        }
        assert_eq!(
            engine.process_transcript("Never mind.", &executor).await,
            Ok(Some("Okay.".to_string()))
        );
        assert!(inner.recorded_transcripts().is_empty());
    }

    #[tokio::test]
    async fn a_phrase_inside_a_longer_utterance_is_not_routed() {
        let (inner, engine) = router();
        let executor = MockToolExecutor::new();
        for said in ["never mind the build", "cancel the build", "nothing yet?"] {
            let reply = engine.process_turns(&[user(said)], &executor).await;
            assert_eq!(reply, Ok(Some("From the engine.".to_string())), "{said:?}");
        }
        assert_eq!(
            inner.recorded_transcripts(),
            vec!["never mind the build", "cancel the build", "nothing yet?"]
        );
    }

    #[tokio::test]
    async fn only_the_latest_user_message_is_matched() {
        let (inner, engine) = router();
        let executor = MockToolExecutor::new();
        // An earlier "cancel" does not route a later request.
        let turns = [user("cancel"), user("list the panes")];
        let reply = engine.process_turns(&turns, &executor).await;
        assert_eq!(reply, Ok(Some("From the engine.".to_string())));
        // Context after the user's message does not hide it.
        let mut context = user("[herdr context]");
        context.role = "system".to_string();
        let turns = [user("never mind"), context];
        let reply = engine.process_turns(&turns, &executor).await;
        assert_eq!(reply, Ok(Some("Okay.".to_string())));
        assert_eq!(inner.recorded_transcripts(), vec!["list the panes"]);
    }

    #[tokio::test]
    async fn a_routed_turn_calls_no_tool_and_not_the_inner_engine() {
        struct CallsATool;
        #[async_trait]
        impl MockHandler for CallsATool {
            async fn handle(
                &self,
                _transcript: &str,
                executor: &dyn ToolExecutor,
            ) -> Result<Option<String>, String> {
                executor
                    .execute_tool("run_command", &serde_json::json!({"command": "continue"}))
                    .await?;
                Ok(Some("Sent.".to_string()))
            }
        }
        let inner = std::sync::Arc::new(MockReasoningEngine::with_handler(CallsATool));
        let engine =
            RoutingReasoningEngine::new(inner.clone(), vec![Route::new(&["never mind"], "Okay.")]);
        let executor = MockToolExecutor::new();

        let reply = engine
            .process_turns(&[user("Never mind.")], &executor)
            .await;
        assert_eq!(reply, Ok(Some("Okay.".to_string())));
        assert!(inner.recorded_transcripts().is_empty());
        assert!(inner.recorded_tool_calls().is_empty());
        assert!(executor.recorded_calls().is_empty());

        // The same engine still reaches its tools for anything else.
        let reply = engine.process_turns(&[user("continue")], &executor).await;
        assert_eq!(reply, Ok(Some("Sent.".to_string())));
        assert_eq!(executor.recorded_calls().len(), 1);
    }

    #[tokio::test]
    async fn the_routed_reply_is_appended_to_the_session() {
        // The caller's side of a turn: load, add the user's message, run
        // the engine, and append both.
        let dir = tempdir().unwrap();
        let manager = SessionManager::with_dir(dir.path().to_path_buf()).unwrap();
        let (_, engine) = router();
        let executor = MockToolExecutor::new();

        let mut turns = manager.load_session("s").unwrap();
        turns.push(user("Forget it."));
        let reply = engine.process_turns(&turns, &executor).await.unwrap();
        manager.append_message("s", &user("Forget it.")).unwrap();
        let model = Message {
            role: "model".to_string(),
            content: reply.unwrap(),
            name: None,
            thought_signature: None,
        };
        manager.append_message("s", &model).unwrap();

        let history = manager.load_session("s").unwrap();
        assert_eq!(history, vec![user("Forget it."), model]);
        assert_eq!(history[1].content, "Okay.");
    }

    #[test]
    fn the_first_route_wins_a_shared_phrase() {
        let inner = std::sync::Arc::new(MockReasoningEngine::new(None));
        let engine = RoutingReasoningEngine::new(
            inner,
            vec![
                Route::new(&["stop"], "First."),
                Route::new(&["Stop!"], "Second."),
            ],
        );
        assert_eq!(engine.route("stop"), Some("First."));
        assert_eq!(engine.route("stop it"), None);
    }
}
