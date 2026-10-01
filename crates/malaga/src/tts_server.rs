//! HTTP server for a VITS / MMS-TTS model (`malaga serve` on a `vits` GGUF).
//!
//! * `POST /v1/audio/speech`  OpenAI compatible: `{input, speed?, response_format?}`
//!   → `audio/wav` (16-bit mono) or raw `pcm` (s16le). `voice` is accepted and
//!   ignored (single-speaker model). Extra knobs: `noise_scale`,
//!   `noise_scale_duration`, `seed`.
//! * `POST /v1/audio/normalize`  `{input}` → `{text}`: what the model will read
//!   (numbers, units, symbols, foreign words spelled out). `normalize: false`
//!   on /v1/audio/speech reads the input as is.
//! * `GET /health`, `GET /v1/models`
//!
//! One inference thread owns the model; requests are served in arrival order.

use std::sync::mpsc;

use anyhow::Result;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use malaga_core::vits::wav_bytes;
use malaga_core::{SynthOptions, Vits};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::oneshot;

pub struct TtsServerConfig {
    pub addr: String,
    pub model_id: Option<String>,
    pub defaults: SynthOptions,
}

enum Job {
    Speak { text: String, opts: SynthOptions, reply: oneshot::Sender<Result<Vec<f32>, String>> },
    Normalize { text: String, opts: SynthOptions, reply: oneshot::Sender<String> },
}

#[derive(Clone)]
struct AppState {
    jobs: mpsc::Sender<Job>,
    model: String,
    defaults: SynthOptions,
    sample_rate: u32,
}

#[derive(Deserialize)]
struct SpeechRequest {
    input: String,
    #[serde(default)]
    speed: Option<f32>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    noise_scale: Option<f32>,
    #[serde(default)]
    noise_scale_duration: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    normalize: Option<bool>,
}

#[derive(Deserialize)]
struct NormalizeRequest {
    input: String,
}

async fn normalize(State(s): State<AppState>, body: axum::body::Bytes) -> Response {
    let req: NormalizeRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")),
    };
    let (reply, rx) = oneshot::channel();
    if s.jobs.send(Job::Normalize { text: req.input, opts: s.defaults.clone(), reply }).is_err() {
        return error(StatusCode::INTERNAL_SERVER_ERROR, "inference thread stopped");
    }
    match rx.await {
        Ok(text) => Json(json!({ "text": text })).into_response(),
        Err(_) => error(StatusCode::INTERNAL_SERVER_ERROR, "inference thread stopped"),
    }
}

fn error(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({ "error": { "message": msg.into(), "type": "invalid_request_error" } }))).into_response()
}

async fn speech(State(s): State<AppState>, body: axum::body::Bytes) -> Response {
    let req: SpeechRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")),
    };
    if req.input.trim().is_empty() {
        return error(StatusCode::BAD_REQUEST, "`input` is empty");
    }
    let format = req.response_format.as_deref().unwrap_or("wav");
    if !matches!(format, "wav" | "pcm") {
        return error(StatusCode::BAD_REQUEST, format!("response_format '{format}' not supported (wav, pcm)"));
    }
    let mut opts = s.defaults.clone();
    if let Some(v) = req.speed {
        if !(0.25..=4.0).contains(&v) {
            return error(StatusCode::BAD_REQUEST, "`speed` must be between 0.25 and 4.0");
        }
        opts.speaking_rate = v;
    }
    opts.noise_scale = req.noise_scale.unwrap_or(opts.noise_scale);
    opts.noise_scale_duration = req.noise_scale_duration.unwrap_or(opts.noise_scale_duration);
    opts.seed = req.seed.or(opts.seed);
    opts.normalize = req.normalize.unwrap_or(opts.normalize);

    let (reply, rx) = oneshot::channel();
    if s.jobs.send(Job::Speak { text: req.input, opts, reply }).is_err() {
        return error(StatusCode::INTERNAL_SERVER_ERROR, "inference thread stopped");
    }
    let samples = match rx.await {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
        Err(_) => return error(StatusCode::INTERNAL_SERVER_ERROR, "inference thread stopped"),
    };
    if samples.is_empty() {
        return error(StatusCode::BAD_REQUEST, "nothing to pronounce in `input` (unsupported characters only)");
    }
    let (body, ctype) = if format == "pcm" {
        let pcm: Vec<u8> =
            samples.iter().flat_map(|x| ((x.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes()).collect();
        (pcm, "audio/pcm")
    } else {
        (wav_bytes(&samples, s.sample_rate), "audio/wav")
    };
    ([(header::CONTENT_TYPE, ctype)], body).into_response()
}

fn worker(v: Vits, rx: mpsc::Receiver<Job>) {
    while let Ok(job) = rx.recv() {
        match job {
            Job::Speak { text, opts, reply } => {
                let t0 = std::time::Instant::now();
                let out = v.synthesize(&text, &opts).map_err(|e| format!("{e:#}"));
                if let Ok(w) = &out {
                    let secs = w.len() as f64 / v.sample_rate() as f64;
                    tracing::info!("{} chars -> {secs:.2}s audio in {:.0?}", text.chars().count(), t0.elapsed());
                }
                let _ = reply.send(out);
            }
            Job::Normalize { text, opts, reply } => {
                let _ = reply.send(v.spoken_text(&text, &opts));
            }
        }
    }
}

pub fn router(v: Vits, cfg: &TtsServerConfig) -> Result<Router> {
    let (tx, rx) = mpsc::channel::<Job>();
    let state = AppState {
        jobs: tx,
        model: cfg.model_id.clone().unwrap_or_else(|| v.name().to_string()),
        defaults: cfg.defaults.clone(),
        sample_rate: v.sample_rate(),
    };
    std::thread::Builder::new().name("malaga-tts".into()).spawn(move || worker(v, rx))?;
    Ok(Router::new()
        .route("/", get(|| async { "malaga: Malagasy text-to-speech server (MMS-TTS / VITS)\n" }))
        .route("/health", get(|| async { Json(json!({ "status": "ok" })) }))
        .route(
            "/v1/models",
            get(|State(s): State<AppState>| async move {
                Json(json!({ "object": "list", "data": [{ "id": s.model, "object": "model", "owned_by": "malaga" }] }))
            }),
        )
        .route("/v1/audio/speech", post(speech))
        .route("/v1/audio/normalize", post(normalize))
        .with_state(state))
}

pub async fn run(v: Vits, cfg: TtsServerConfig) -> Result<()> {
    let app = router(v, &cfg)?;
    let listener = tokio::net::TcpListener::bind(&cfg.addr).await?;
    tracing::info!("listening on http://{}", cfg.addr);
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
