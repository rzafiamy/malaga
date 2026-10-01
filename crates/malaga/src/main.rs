mod server;

use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use malaga_core::convert::{convert, ConvertOptions, Preset};
use malaga_core::{best_device, Device, GenOptions, Translator};

#[derive(Parser)]
#[command(name = "malaga", version, about = "FR/EN -> Malagasy translation with NLLB-200 on GGUF")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Convert a Hugging Face NLLB checkpoint directory to GGUF.
    Convert {
        /// Directory with config.json, tokenizer.json and the weights.
        #[arg(long)]
        hf: PathBuf,
        #[arg(long, short)]
        out: PathBuf,
        /// f32, f16, q8_0, q6_k, q5_k_m, q4_k_m
        #[arg(long, default_value = "q8_0")]
        preset: Preset,
        /// Hugging Face repo id, stored in the GGUF metadata.
        #[arg(long, default_value = "facebook/nllb-200-distilled-600M")]
        name: String,
        /// Vocabulary shortlist for a target language, built from a text corpus:
        /// `--shortlist mg=corpus.txt`. Speeds up GPU decoding to that language.
        #[arg(long, value_parser = parse_shortlist)]
        shortlist: Vec<(String, PathBuf)>,
        /// Minimum corpus frequency for a token to enter a shortlist.
        #[arg(long, default_value_t = 2)]
        shortlist_min_count: usize,
    },
    /// Translate text given as argument or on stdin.
    Translate {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        gen: GenArgs,
        #[arg(long, short = 'f', default_value = "fr")]
        from: String,
        #[arg(long, short = 't', default_value = "mg")]
        to: String,
        /// Treat every input line as one segment (no sentence splitting) and
        /// translate all lines in batches. Handy for evaluation and bulk jobs.
        #[arg(long)]
        lines: bool,
        /// Text to translate (reads stdin when absent).
        text: Option<String>,
    },
    /// Measure latency / throughput.
    Bench {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        gen: GenArgs,
        #[arg(long, default_value_t = 20)]
        iters: usize,
        /// Sentences per batch for the throughput test.
        #[arg(long, default_value_t = 16)]
        batch: usize,
    },
    /// Run the HTTP server (native, OpenAI and Ollama compatible APIs).
    Serve {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        gen: GenArgs,
        /// Model name reported by the API (defaults to the GGUF `general.name`).
        #[arg(long)]
        model_id: Option<String>,
        #[arg(long, env = "MALAGA_HOST", default_value = "127.0.0.1")]
        host: String,
        #[arg(long, env = "MALAGA_PORT", default_value_t = 8080)]
        port: u16,
        /// Source language when a request does not specify one.
        #[arg(long, default_value = "fr")]
        default_source: String,
        /// Target language when a request does not specify one.
        #[arg(long, default_value = "mg")]
        default_target: String,
    },
}

#[derive(Args)]
struct ModelArgs {
    /// Path to the GGUF model.
    #[arg(long, short, env = "MALAGA_MODEL")]
    model: PathBuf,
    /// auto (CUDA, then Metal, then CPU), cpu, cuda or metal.
    #[arg(long, default_value = "auto", env = "MALAGA_DEVICE")]
    device: String,
    /// GPU ordinal.
    #[arg(long, default_value_t = 0)]
    device_id: usize,
    /// CPU threads (CPU inference and tokenization).
    #[arg(long)]
    threads: Option<usize>,
}

impl ModelArgs {
    fn device(&self) -> Result<Device> {
        Ok(match self.device.as_str() {
            "auto" => best_device(self.device_id)?,
            "cpu" => Device::Cpu,
            "cuda" | "gpu" => Device::new_cuda(self.device_id)?,
            "metal" => Device::new_metal(self.device_id)?,
            d => anyhow::bail!("unknown device '{d}' (auto, cpu, cuda, metal)"),
        })
    }

    fn load(&self) -> Result<Translator> {
        if let Some(n) = self.threads {
            // Read by candle's CPU backend and by rayon (tokenizer).
            std::env::set_var("RAYON_NUM_THREADS", n.to_string());
        }
        let device = self.device()?;
        let t0 = Instant::now();
        let t = Translator::load(&self.model, &device)?;
        tracing::info!("loaded {} on {device:?} in {:.2?}", t.name(), t0.elapsed());
        Ok(t)
    }
}

#[derive(Args, Clone)]
struct GenArgs {
    /// Beam size (1 = greedy, fastest).
    #[arg(long, default_value_t = 1)]
    beam: usize,
    #[arg(long)]
    max_new_tokens: Option<usize>,
    /// Maximum sentences decoded together.
    #[arg(long, default_value_t = 32)]
    max_batch: usize,
    /// Speed mode: decode over the target language's vocabulary shortlist
    /// (if the model has one). Faster and ~200 MB less VRAM, but not exact:
    /// rare tokens missing from the shortlist can change a few sentences.
    #[arg(long)]
    fast_vocab: bool,
    /// Use the generic (unfused) GPU implementation, the reference the fused
    /// path is validated against. Slower; for verification.
    #[arg(long)]
    reference: bool,
}

impl GenArgs {
    fn options(&self) -> GenOptions {
        GenOptions {
            beam_size: self.beam,
            max_new_tokens: self.max_new_tokens,
            max_batch: self.max_batch,
            shortlist: self.fast_vocab,
            fused: !self.reference,
            ..GenOptions::default()
        }
    }
}

const BENCH_SENTENCES: &[&str] = &[
    "Bonjour, comment allez-vous aujourd'hui ?",
    "Le gouvernement a annoncé de nouvelles mesures pour soutenir les agriculteurs de la région.",
    "Madagascar est une grande île située dans l'océan Indien, au large de la côte est de l'Afrique.",
    "Les enfants vont à l'école tous les matins à sept heures.",
    "Il pleut beaucoup pendant la saison des pluies, de novembre à mars.",
    "Nous devons protéger les forêts et la biodiversité unique de notre pays.",
    "Le marché ouvre très tôt et les vendeurs installent leurs étals avant le lever du soleil.",
    "Merci beaucoup pour votre aide.",
];

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with_writer(std::io::stderr)
        .init();

    match Cli::parse().cmd {
        Cmd::Convert { hf, out, preset, name, shortlist, shortlist_min_count } => {
            let t0 = Instant::now();
            let opts = ConvertOptions { preset, model_name: name, shortlists: shortlist, shortlist_min_count };
            convert(&hf, &out, &opts)?;
            tracing::info!("done in {:.1?}", t0.elapsed());
        }
        Cmd::Translate { model, gen, from, to, lines, text } => {
            let t = model.load()?;
            let text = match text {
                Some(t) => t,
                None => {
                    let mut s = String::new();
                    std::io::stdin().read_to_string(&mut s)?;
                    s
                }
            };
            let t0 = Instant::now();
            if lines {
                let input: Vec<&str> = text.lines().collect();
                for line in t.translate_batch(&input, &from, &to, &gen.options())? {
                    println!("{line}");
                }
            } else {
                println!("{}", t.translate(text.trim_end(), &from, &to, &gen.options())?);
            }
            tracing::info!("translated in {:.2?}", t0.elapsed());
        }
        Cmd::Bench { model, gen, iters, batch } => bench(&model.load()?, &gen.options(), iters, batch)?,
        Cmd::Serve { model, gen, model_id, host, port, default_source, default_target } => {
            let translator = model.load()?;
            // Warm up (CUDA module JIT, graph capture for the common shape) before
            // the port opens, so that a readiness probe on /health covers it.
            let t0 = Instant::now();
            translator.translate("Bonjour.", "fr", &default_target, &gen.options())?;
            tracing::info!("warm-up done in {:.2?}", t0.elapsed());
            let cfg = server::ServerConfig {
                addr: format!("{host}:{port}"),
                model_id,
                defaults: gen.options(),
                default_source,
                default_target,
            };
            tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(server::run(translator, cfg))?;
        }
    }
    Ok(())
}

fn parse_shortlist(s: &str) -> Result<(String, PathBuf), String> {
    let (lang, path) = s.split_once('=').ok_or("expected LANG=CORPUS")?;
    Ok((lang.to_string(), PathBuf::from(path)))
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn bench(t: &Translator, opts: &GenOptions, iters: usize, batch: usize) -> Result<()> {
    let s = BENCH_SENTENCES[1];
    // Warm-up (CUDA kernels, allocator).
    for _ in 0..3 {
        t.translate_batch(&[s], "fr", "mg", opts)?;
    }

    let ids = t.encode(s, t.lang_id("fr")?)?;
    let tgt = t.lang_id("mg")?;
    let mut lat = Vec::with_capacity(iters);
    let mut tokens = 0usize;
    for _ in 0..iters {
        let t0 = Instant::now();
        tokens = t.generate(std::slice::from_ref(&ids), tgt, opts)?[0].len();
        lat.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let p50 = percentile(&mut lat, 0.5);
    let enc = {
        let t0 = Instant::now();
        for _ in 0..iters {
            t.model().encode(std::slice::from_ref(&ids), 1)?.batch_size()?;
        }
        t.model().device().synchronize()?;
        t0.elapsed().as_secs_f64() * 1e3 / iters as f64
    };
    println!("single sentence: {} source tokens -> {tokens} target tokens", ids.len());
    println!("  latency p50 {p50:.1} ms | p90 {:.1} ms | min {:.1} ms", percentile(&mut lat, 0.9), lat[0]);
    println!("  encoder {enc:.2} ms | decoder {:.2} ms/token", (p50 - enc) / (tokens + 1) as f64);

    let sentences: Vec<&str> = BENCH_SENTENCES.iter().cycle().take(batch).copied().collect();
    let t0 = Instant::now();
    let rounds = (iters / 4).max(1);
    let mut chars = 0;
    for _ in 0..rounds {
        let out = t.translate_batch(&sentences, "fr", "mg", opts)?;
        chars += out.iter().map(|s| s.len()).sum::<usize>();
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "batch of {batch}: {:.1} ms/batch | {:.1} sentences/s | {:.0} chars/s",
        dt * 1e3 / rounds as f64,
        (batch * rounds) as f64 / dt,
        chars as f64 / dt
    );
    if let Some(mib) = gpu_memory_mib() {
        println!("GPU memory used by this process: {mib} MiB");
    }
    Ok(())
}

/// VRAM used by the current process, as reported by `nvidia-smi`.
fn gpu_memory_mib() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    let pid = std::process::id().to_string();
    String::from_utf8_lossy(&out.stdout).lines().find_map(|l| {
        let (p, m) = l.split_once(',')?;
        (p.trim() == pid).then(|| m.trim().parse().ok())?
    })
}
