//! HTTP server.
//!
//! * `POST /v1/translate`         native API
//! * `POST /v1/chat/completions`  OpenAI compatible (JSON or SSE `stream: true`)
//! * `POST /api/generate`, `POST /api/chat`, `GET /api/tags`  Ollama compatible
//!
//! All requests go through a single inference thread that owns the model.
//! Requests that arrive while the GPU is busy are merged into one batch
//! (dynamic batching), so concurrency increases throughput without adding any
//! latency to a lone request.

use std::sync::mpsc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use malaga_core::{GenOptions, Translator};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::oneshot;

pub struct ServerConfig {
    pub addr: String,
    pub model_id: Option<String>,
    pub defaults: GenOptions,
    pub default_source: String,
    pub default_target: String,
}

struct Job {
    texts: Vec<String>,
    source: String,
    target: String,
    beam: usize,
    reply: oneshot::Sender<Result<Vec<String>, String>>,
}

#[derive(Clone)]
struct AppState {
    jobs: mpsc::Sender<Job>,
    model: String,
    default_source: String,
    default_target: String,
    default_beam: usize,
}

impl AppState {
    async fn translate(
        &self,
        texts: Vec<String>,
        source: Option<String>,
        target: Option<String>,
        beam: Option<usize>,
    ) -> Result<Vec<String>, ApiError> {
        let (reply, rx) = oneshot::channel();
        let job = Job {
            texts,
            source: source.unwrap_or_else(|| self.default_source.clone()),
            target: target.unwrap_or_else(|| self.default_target.clone()),
            beam: beam.unwrap_or(self.default_beam).clamp(1, 8),
            reply,
        };
        self.jobs.send(job).map_err(|_| ApiError::internal("inference thread stopped"))?;
        rx.await.map_err(|_| ApiError::internal("inference thread dropped the request"))?.map_err(ApiError::bad_request)
    }
}

/// Inference loop: takes every job already queued, groups compatible ones and
/// runs each group as a single batched translation.
fn worker(t: Translator, rx: mpsc::Receiver<Job>, defaults: GenOptions) {
    while let Ok(first) = rx.recv() {
        let mut jobs = vec![first];
        while let Ok(j) = rx.try_recv() {
            jobs.push(j);
        }
        while !jobs.is_empty() {
            let key = (jobs[0].source.clone(), jobs[0].target.clone(), jobs[0].beam);
            let (group, rest): (Vec<Job>, Vec<Job>) =
                jobs.into_iter().partition(|j| (j.source.clone(), j.target.clone(), j.beam) == key);
            jobs = rest;
            let opts = GenOptions { beam_size: key.2, ..defaults.clone() };
            run_group(&t, group, &key.0, &key.1, &opts);
        }
    }
}

fn run_group(t: &Translator, group: Vec<Job>, src: &str, tgt: &str, opts: &GenOptions) {
    let t0 = Instant::now();
    // Every text of every job becomes a document; documents are split into
    // sentences and all sentences of the group are decoded together.
    let docs: Vec<&str> = group.iter().flat_map(|j| j.texts.iter().map(|s| s.as_str())).collect();
    let result = t.translate_docs(&docs, src, tgt, opts).map_err(|e| e.to_string());
    tracing::debug!("batch of {} docs ({} jobs) in {:.1?}", docs.len(), group.len(), t0.elapsed());
    match result {
        Ok(out) => {
            let mut out = out.into_iter();
            for j in group {
                let n = j.texts.len();
                let _ = j.reply.send(Ok(out.by_ref().take(n).collect()));
            }
        }
        Err(e) => {
            for j in group {
                let _ = j.reply.send(Err(e.clone()));
            }
        }
    }
}

struct ApiError(StatusCode, String);

impl ApiError {
    fn bad_request(m: impl Into<String>) -> Self {
        Self(StatusCode::BAD_REQUEST, m.into())
    }
    fn internal(m: impl Into<String>) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, m.into())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": { "message": self.1 } }))).into_response()
    }
}

/// JSON body without a `Content-Type` requirement (Ollama clients often omit it).
struct LooseJson<T>(T);

impl<T: serde::de::DeserializeOwned, S: Send + Sync> axum::extract::FromRequest<S> for LooseJson<T> {
    type Rejection = ApiError;

    async fn from_request(req: axum::extract::Request, state: &S) -> Result<Self, Self::Rejection> {
        let body =
            axum::body::Bytes::from_request(req, state).await.map_err(|e| ApiError::bad_request(e.to_string()))?;
        serde_json::from_slice(&body).map(LooseJson).map_err(|e| ApiError::bad_request(e.to_string()))
    }
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Accepts a single string or a list of strings.
#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct TranslateRequest {
    #[serde(alias = "q", alias = "texts")]
    text: OneOrMany,
    #[serde(alias = "source_lang", alias = "from")]
    source: Option<String>,
    #[serde(alias = "target_lang", alias = "to")]
    target: Option<String>,
    beam_size: Option<usize>,
}

#[derive(Serialize)]
struct TranslateResponse {
    translation: serde_json::Value,
    source: String,
    target: String,
    model: String,
    elapsed_ms: f64,
}

async fn translate(
    State(s): State<AppState>,
    Json(req): Json<TranslateRequest>,
) -> Result<Json<TranslateResponse>, ApiError> {
    let t0 = Instant::now();
    let (texts, single) = match req.text {
        OneOrMany::One(t) => (vec![t], true),
        OneOrMany::Many(v) => (v, false),
    };
    let source = req.source.clone().unwrap_or_else(|| s.default_source.clone());
    let target = req.target.clone().unwrap_or_else(|| s.default_target.clone());
    let out = s.translate(texts, Some(source.clone()), Some(target.clone()), req.beam_size).await?;
    let translation = if single { json!(out[0]) } else { json!(out) };
    Ok(Json(TranslateResponse {
        translation,
        source,
        target,
        model: s.model.clone(),
        elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
    }))
}

/// Languages can be given explicitly, or encoded in the model name:
/// `malaga:fr-mg`, `nllb-en-mg`, ...
fn langs_from_model(model: Option<&str>) -> (Option<String>, Option<String>) {
    let Some(m) = model else { return (None, None) };
    let tail = m.rsplit([':', '/']).next().unwrap_or(m);
    let parts: Vec<&str> = tail.rsplitn(3, '-').collect();
    match parts.as_slice() {
        [tgt, src, ..] if (2..=3).contains(&tgt.len()) && (2..=3).contains(&src.len()) => {
            (Some(src.to_string()), Some(tgt.to_string()))
        }
        _ => (None, None),
    }
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(alias = "source_lang")]
    source: Option<String>,
    #[serde(alias = "target_lang")]
    target: Option<String>,
    #[serde(default)]
    stream: bool,
}

fn last_user(messages: &[ChatMessage]) -> Result<String, ApiError> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .ok_or_else(|| ApiError::bad_request("no user message"))
}

async fn chat_completions(State(s): State<AppState>, Json(req): Json<ChatRequest>) -> Result<Response, ApiError> {
    let text = last_user(&req.messages)?;
    let (ms, mt) = langs_from_model(req.model.as_deref());
    let out = s.translate(vec![text], req.source.or(ms), req.target.or(mt), None).await?;
    let (id, created) = (format!("malaga-{}", now()), now());
    let model = req.model.unwrap_or_else(|| s.model.clone());
    if req.stream {
        // A translation is produced in one go: stream it as a single delta,
        // then the final chunk and `[DONE]` (OpenAI SSE framing).
        let chunk = |delta: serde_json::Value, finish: Option<&str>| {
            json!({ "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] })
        };
        let body = format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            chunk(json!({ "role": "assistant", "content": out[0] }), None),
            chunk(json!({}), Some("stop"))
        );
        return Ok(([("content-type", "text/event-stream"), ("cache-control", "no-cache")], body).into_response());
    }
    Ok(Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": out[0] },
            "finish_reason": "stop"
        }],
    }))
    .into_response())
}

#[derive(Deserialize)]
struct OllamaGenerate {
    model: Option<String>,
    prompt: String,
    #[serde(alias = "source_lang")]
    source: Option<String>,
    #[serde(alias = "target_lang")]
    target: Option<String>,
    #[serde(default = "yes")]
    stream: bool,
}

fn yes() -> bool {
    true
}

/// Ollama streams NDJSON by default; we answer with a single final chunk.
fn ollama_reply(stream: bool, body: serde_json::Value) -> Response {
    if stream {
        ([("content-type", "application/x-ndjson")], format!("{body}\n")).into_response()
    } else {
        Json(body).into_response()
    }
}

async fn ollama_generate(
    State(s): State<AppState>,
    LooseJson(req): LooseJson<OllamaGenerate>,
) -> Result<Response, ApiError> {
    let t0 = Instant::now();
    let (ms, mt) = langs_from_model(req.model.as_deref());
    let out = s.translate(vec![req.prompt], req.source.or(ms), req.target.or(mt), None).await?;
    Ok(ollama_reply(
        req.stream,
        json!({
            "model": req.model.unwrap_or_else(|| s.model.clone()),
            "created_at": now(),
            "response": out[0],
            "done": true,
            "done_reason": "stop",
            "total_duration": t0.elapsed().as_nanos() as u64,
        }),
    ))
}

async fn ollama_chat(State(s): State<AppState>, LooseJson(req): LooseJson<ChatRequest>) -> Result<Response, ApiError> {
    let t0 = Instant::now();
    let text = last_user(&req.messages)?;
    let (ms, mt) = langs_from_model(req.model.as_deref());
    let out = s.translate(vec![text], req.source.or(ms), req.target.or(mt), None).await?;
    Ok(ollama_reply(
        req.stream,
        json!({
            "model": req.model.unwrap_or_else(|| s.model.clone()),
            "created_at": now(),
            "message": { "role": "assistant", "content": out[0] },
            "done": true,
            "done_reason": "stop",
            "total_duration": t0.elapsed().as_nanos() as u64,
        }),
    ))
}

async fn models(State(s): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({ "object": "list", "data": [{ "id": s.model, "object": "model", "owned_by": "malaga" }] }))
}

async fn tags(State(s): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({ "models": [{ "name": s.model, "model": s.model, "details": { "family": "nllb" } }] }))
}

/// Builds the HTTP API and starts the inference thread that owns `translator`.
pub fn router(translator: Translator, cfg: &ServerConfig) -> Result<Router> {
    let (tx, rx) = mpsc::channel::<Job>();
    let state = AppState {
        jobs: tx,
        model: cfg.model_id.clone().unwrap_or_else(|| translator.name().to_string()),
        default_source: cfg.default_source.clone(),
        default_target: cfg.default_target.clone(),
        default_beam: cfg.defaults.beam_size,
    };
    let defaults = cfg.defaults.clone();
    std::thread::Builder::new().name("malaga-inference".into()).spawn(move || worker(translator, rx, defaults))?;

    Ok(Router::new()
        .route("/", get(|| async { "malaga: FR/EN -> Malagasy translation server\n" }))
        .route("/health", get(|| async { Json(json!({ "status": "ok" })) }))
        .route("/v1/translate", post(translate))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(models))
        .route("/api/generate", post(ollama_generate))
        .route("/api/chat", post(ollama_chat))
        .route("/api/tags", get(tags))
        .with_state(state))
}

pub async fn run(translator: Translator, cfg: ServerConfig) -> Result<()> {
    let app = router(translator, &cfg)?;
    let listener = tokio::net::TcpListener::bind(&cfg.addr).await?;
    tracing::info!("listening on http://{}", cfg.addr);
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use malaga_core::convert::{convert, ConvertOptions, Preset};
    use malaga_core::Device;
    use tower::ServiceExt;

    /// covers: REQ-SRV-003
    #[test]
    fn model_name_langs() {
        let s = |a: &str, b: &str| (Some(a.to_string()), Some(b.to_string()));
        assert_eq!(langs_from_model(Some("malaga:fr-mg")), s("fr", "mg"));
        assert_eq!(langs_from_model(Some("nllb-600m-en-mg")), s("en", "mg"));
        assert_eq!(langs_from_model(Some("malaga")), (None, None));
        assert_eq!(langs_from_model(None), (None, None));
    }

    fn app(dir: &std::path::Path) -> Router {
        let hf = dir.join("hf");
        malaga_core::testutil::write_tiny_checkpoint(&hf, 3).unwrap();
        let gguf = dir.join("tiny.gguf");
        let opts = ConvertOptions {
            preset: Preset::Q8_0,
            model_name: "tiny".into(),
            shortlists: vec![],
            shortlist_min_count: 1,
        };
        convert(&hf, &gguf, &opts).unwrap();
        let t = Translator::load(&gguf, &Device::Cpu).unwrap();
        let cfg = ServerConfig {
            addr: String::new(),
            model_id: Some("malaga-test".into()),
            defaults: GenOptions::default(),
            default_source: "fr".into(),
            default_target: "mg".into(),
        };
        router(t, &cfg).unwrap()
    }

    async fn call(
        app: &Router,
        method: &str,
        uri: &str,
        body: Option<&str>,
        json_ct: bool,
    ) -> (StatusCode, serde_json::Value) {
        let mut req = Request::builder().method(method).uri(uri);
        if json_ct {
            req = req.header("content-type", "application/json");
        }
        let req = req.body(body.map(|b| Body::from(b.to_string())).unwrap_or_else(Body::empty)).unwrap();
        let res = app.clone().oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 20).await.unwrap();
        let text = String::from_utf8_lossy(&bytes);
        let first = text.lines().next().unwrap_or("null");
        let first = first.strip_prefix("data: ").unwrap_or(first);
        (status, serde_json::from_str(first).unwrap_or(serde_json::Value::Null))
    }

    /// covers: REQ-SRV-001, REQ-SRV-002, REQ-SRV-003, REQ-SRV-004, REQ-SRV-005
    #[tokio::test(flavor = "multi_thread")]
    async fn every_api_answers() {
        let dir = tempfile::tempdir().unwrap();
        let app = app(dir.path());
        let (st, v) = call(&app, "GET", "/health", None, false).await;
        assert_eq!((st, v["status"].as_str()), (StatusCode::OK, Some("ok")));

        let (st, v) = call(&app, "POST", "/v1/translate", Some(r#"{"text":"w1 w2"}"#), true).await;
        assert_eq!(st, StatusCode::OK);
        assert!(v["translation"].is_string());
        assert_eq!((v["source"].as_str(), v["target"].as_str()), (Some("fr"), Some("mg")));

        let (_, v) = call(&app, "POST", "/v1/translate", Some(r#"{"texts":["w1","w2 w3"],"from":"en"}"#), true).await;
        assert_eq!(v["translation"].as_array().map(|a| a.len()), Some(2));

        let body = r#"{"model":"malaga:fr-mg","messages":[{"role":"user","content":"w4"}]}"#;
        let (_, v) = call(&app, "POST", "/v1/chat/completions", Some(body), true).await;
        assert!(v["choices"][0]["message"]["content"].is_string());

        let stream = r#"{"model":"malaga:fr-mg","stream":true,"messages":[{"role":"user","content":"w4"}]}"#;
        let (st, v) = call(&app, "POST", "/v1/chat/completions", Some(stream), true).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(v["object"].as_str(), Some("chat.completion.chunk"));

        // Ollama clients may omit the content type and default to streaming.
        let (_, v) =
            call(&app, "POST", "/api/generate", Some(r#"{"model":"malaga:en-mg","prompt":"w5"}"#), false).await;
        assert_eq!(v["done"].as_bool(), Some(true));
        let (_, v) = call(&app, "POST", "/api/chat", Some(body), false).await;
        assert!(v["message"]["content"].is_string());

        let (_, v) = call(&app, "GET", "/v1/models", None, false).await;
        assert_eq!(v["data"][0]["id"].as_str(), Some("malaga-test"));
        let (_, v) = call(&app, "GET", "/api/tags", None, false).await;
        assert_eq!(v["models"][0]["name"].as_str(), Some("malaga-test"));
    }

    /// covers: REQ-SRV-006
    #[tokio::test(flavor = "multi_thread")]
    async fn bad_requests_are_400_with_a_message() {
        let dir = tempfile::tempdir().unwrap();
        let app = app(dir.path());
        let (st, v) = call(&app, "POST", "/v1/translate", Some(r#"{"text":"w1","target":"klingon"}"#), true).await;
        assert_eq!(st, StatusCode::BAD_REQUEST);
        assert!(v["error"]["message"].as_str().unwrap().contains("klingon"));
        let (st, _) = call(&app, "POST", "/api/generate", Some("not json"), false).await;
        assert_eq!(st, StatusCode::BAD_REQUEST);
    }

    /// Concurrent requests are merged into one batch and each gets its own answer.
    /// covers: REQ-SRV-007
    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_requests_get_their_own_translation() {
        let dir = tempfile::tempdir().unwrap();
        let app = app(dir.path());
        let texts: Vec<String> = (0..8).map(|i| malaga_core::testutil::sentence(2 + i, i)).collect();
        let single: Vec<serde_json::Value> = {
            let mut v = vec![];
            for t in &texts {
                v.push(call(&app, "POST", "/v1/translate", Some(&json!({ "text": t }).to_string()), true).await.1);
            }
            v
        };
        let handles: Vec<_> = texts
            .iter()
            .map(|t| {
                let app = app.clone();
                let body = json!({ "text": t }).to_string();
                tokio::spawn(async move { call(&app, "POST", "/v1/translate", Some(&body), true).await.1 })
            })
            .collect();
        let mut together = vec![];
        for h in handles {
            together.push(h.await.unwrap());
        }
        for (a, b) in single.iter().zip(&together) {
            assert_eq!(a["translation"], b["translation"]);
        }
    }
}
