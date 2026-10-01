//! Friendly language names -> NLLB-200 language codes.

/// Languages this project focuses on. Any other NLLB-200 code (e.g. `deu_Latn`)
/// is also accepted as-is by [`resolve`].
pub const FOCUS: &[(&str, &str)] = &[("fr", "fra_Latn"), ("en", "eng_Latn"), ("mg", "plt_Latn")];

/// Resolves `fr`, `fra`, `fr-FR`, `french`, `français`, `mg`, `malagasy`,
/// `malgache`, ... into an NLLB code. Unknown inputs are returned unchanged so
/// that raw NLLB codes keep working.
pub fn resolve(lang: &str) -> String {
    let l = lang.trim();
    let lower = l.to_lowercase();
    let base = lower.split(['-', '_']).next().unwrap_or_default();
    let code = match base {
        "fr" | "fra" | "fre" | "french" | "français" | "francais" => "fra_Latn",
        "en" | "eng" | "english" | "anglais" => "eng_Latn",
        "mg" | "mlg" | "plt" | "malagasy" | "malgache" => "plt_Latn",
        _ => return l.to_string(),
    };
    code.to_string()
}

#[cfg(test)]
mod tests {
    use super::resolve;

    /// covers: REQ-LNG-001
    #[test]
    fn aliases() {
        assert_eq!(resolve("fr"), "fra_Latn");
        assert_eq!(resolve("fr-FR"), "fra_Latn");
        assert_eq!(resolve("Français"), "fra_Latn");
        assert_eq!(resolve("en_US"), "eng_Latn");
        assert_eq!(resolve("MG"), "plt_Latn");
        assert_eq!(resolve("malgache"), "plt_Latn");
        assert_eq!(resolve("plt_Latn"), "plt_Latn");
        assert_eq!(resolve("deu_Latn"), "deu_Latn");
    }
}
