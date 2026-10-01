//! End-to-end tests on a tiny random NLLB checkpoint (no download needed).
//! Requirement ids refer to `spec/specification.md`.

use std::path::{Path, PathBuf};

use malaga_core::convert::{convert, ConvertOptions, Preset};
use malaga_core::testutil::{sentence, write_tiny_checkpoint};
use malaga_core::{Device, GenOptions, Translator};

fn convert_tiny(dir: &Path, preset: Preset, shortlist: Option<&Path>) -> PathBuf {
    let hf = dir.join("hf");
    if !hf.join("config.json").exists() {
        write_tiny_checkpoint(&hf, 7).unwrap();
    }
    let out = dir.join(format!("tiny-{}.gguf", preset.name()));
    let opts = ConvertOptions {
        preset,
        model_name: "tiny".into(),
        shortlists: shortlist.map(|p| vec![("mg".to_string(), p.to_path_buf())]).unwrap_or_default(),
        shortlist_min_count: 1,
    };
    convert(&hf, &out, &opts).unwrap();
    out
}

fn sentences() -> Vec<String> {
    (0..6).map(|i| sentence(3 + 4 * i, i)).collect()
}

/// covers: REQ-CNV-001, REQ-CNV-002, REQ-INF-001
#[test]
fn convert_load_and_translate_every_preset() {
    let dir = tempfile::tempdir().unwrap();
    for preset in [Preset::F32, Preset::F16, Preset::Q8_0, Preset::Q4KM] {
        let path = convert_tiny(dir.path(), preset, None);
        let t = Translator::load(&path, &Device::Cpu).unwrap();
        assert_eq!(t.name(), "tiny");
        assert_eq!(t.model().cfg.vocab_size, 4 + malaga_core::testutil::WORDS + 3);
        let out = t.translate("w1 w2 w3. w4 w5", "fr", "mg", &GenOptions::default()).unwrap();
        assert!(!out.is_empty(), "{preset:?}");
    }
}

/// covers: REQ-INF-002
#[test]
fn batching_does_not_change_translations() {
    let dir = tempfile::tempdir().unwrap();
    let t = Translator::load(convert_tiny(dir.path(), Preset::F32, None), &Device::Cpu).unwrap();
    let s = sentences();
    let refs: Vec<&str> = s.iter().map(|x| x.as_str()).collect();
    let opts = GenOptions::default();
    let batched = t.translate_batch(&refs, "fr", "mg", &opts).unwrap();
    for (src, b) in refs.iter().zip(&batched) {
        let single = t.translate_batch(&[src], "fr", "mg", &opts).unwrap().remove(0);
        assert_eq!(&single, b, "padding must not leak into other rows");
    }
}

/// covers: REQ-INF-003
#[test]
fn beam_search_is_deterministic_and_beam1_is_greedy() {
    let dir = tempfile::tempdir().unwrap();
    let t = Translator::load(convert_tiny(dir.path(), Preset::F32, None), &Device::Cpu).unwrap();
    let s = sentence(8, 3);
    let greedy = t.translate(&s, "fr", "mg", &GenOptions::default()).unwrap();
    let beam = |k| t.translate(&s, "fr", "mg", &GenOptions { beam_size: k, ..Default::default() }).unwrap();
    assert_eq!(beam(1), greedy);
    assert_eq!(beam(3), beam(3));
}

/// covers: REQ-INF-004
#[test]
fn document_layout_is_preserved() {
    let dir = tempfile::tempdir().unwrap();
    let t = Translator::load(convert_tiny(dir.path(), Preset::F32, None), &Device::Cpu).unwrap();
    let out = t.translate("w1 w2.\n\n  w3 w4!\nw5", "fr", "mg", &GenOptions::default()).unwrap();
    assert_eq!(out.matches('\n').count(), 3);
    assert!(out.contains("\n\n  "));
}

/// covers: REQ-CNV-003
#[test]
fn shortlist_keeps_corpus_tokens_and_the_language_token() {
    let dir = tempfile::tempdir().unwrap();
    let corpus = dir.path().join("mg.txt");
    std::fs::write(&corpus, "w1 w2 w3\nw2 w9\n").unwrap();
    let t = Translator::load(convert_tiny(dir.path(), Preset::Q8_0, Some(&corpus)), &Device::Cpu).unwrap();
    let sl = t.model().shortlist("plt_Latn").expect("shortlist stored in the GGUF");
    let lang = t.lang_id("mg").unwrap();
    for id in [0, 1, 2, 3, 5, 6, 7, 13, lang] {
        assert!(sl.contains(id), "token {id} missing");
    }
    assert!(!sl.contains(4 + 100), "unseen token must not be kept");
}

/// covers: REQ-LNG-001
#[test]
fn unknown_language_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let t = Translator::load(convert_tiny(dir.path(), Preset::Q8_0, None), &Device::Cpu).unwrap();
    assert!(t.translate("w1", "fr", "klingon", &GenOptions::default()).is_err());
    assert!(t.lang_id("français").is_ok());
}

/// The fused CUDA path (static buffers, fused kernels, CUDA graphs, f16 KV cache)
/// must pick the same tokens as the generic implementation.
/// covers: REQ-GPU-001
#[cfg(feature = "cuda")]
#[test]
fn fused_gpu_decoding_matches_the_reference_path() {
    let Ok(dev) = Device::new_cuda(0) else { return };
    let dir = tempfile::tempdir().unwrap();
    for preset in [Preset::F32, Preset::Q8_0] {
        let t = Translator::load(convert_tiny(dir.path(), preset, None), &dev).unwrap();
        let s = sentences();
        let refs: Vec<&str> = s.iter().map(|x| x.as_str()).collect();
        let fused = t.translate_batch(&refs, "fr", "mg", &GenOptions::default()).unwrap();
        let reference =
            t.translate_batch(&refs, "fr", "mg", &GenOptions { fused: false, ..Default::default() }).unwrap();
        assert_eq!(fused, reference, "{preset:?}");
        // Second run replays the cached CUDA graphs.
        assert_eq!(t.translate_batch(&refs, "fr", "mg", &GenOptions::default()).unwrap(), fused);
    }
}
