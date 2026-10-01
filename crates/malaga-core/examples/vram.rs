use malaga_core::{Device, GenOptions, Translator};
fn used() -> u64 {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
        .output()
        .unwrap();
    let pid = std::process::id().to_string();
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|l| {
            let (p, m) = l.split_once(',')?;
            (p.trim() == pid).then(|| m.trim().parse().ok())?
        })
        .unwrap_or(0)
}
fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).unwrap();
    let dev = Device::new_cuda(0)?;
    let _ = candle_core::Tensor::zeros(1, candle_core::DType::F32, &dev)?;
    println!("context: {} MiB", used());
    let t = Translator::load(&path, &dev)?;
    dev.synchronize()?;
    println!("loaded: {} MiB", used());
    let o = GenOptions::default();
    t.translate("Bonjour tout le monde, comment allez-vous ?", "fr", "mg", &o)?;
    println!("after 1 sentence: {} MiB", used());
    let long = "Le gouvernement a annoncé de nouvelles mesures pour soutenir les agriculteurs de la région, notamment des aides financières, des formations techniques et un meilleur accès aux marchés, afin de réduire la pauvreté rurale et de protéger les forêts qui abritent une biodiversité unique au monde, tout en améliorant les routes, les écoles et les centres de santé dans les villages les plus isolés du pays.";
    for n in [1usize, 2, 4] {
        let s: Vec<&str> = std::iter::repeat_n(long, n).collect();
        t.translate_batch(&s, "fr", "mg", &o)?;
        println!("after long x{n}: {} MiB", used());
    }
    let s: Vec<&str> = std::iter::repeat_n(
        "Le gouvernement a annoncé de nouvelles mesures pour soutenir les agriculteurs de la région.",
        16,
    )
    .collect();
    t.translate_batch(&s, "fr", "mg", &o)?;
    println!("after batch 16: {} MiB", used());
    let s: Vec<&str> = std::iter::repeat_n(
        "Le gouvernement a annoncé de nouvelles mesures pour soutenir les agriculteurs de la région.",
        32,
    )
    .collect();
    t.translate_batch(&s, "fr", "mg", &o)?;
    println!("after batch 32: {} MiB", used());
    Ok(())
}
