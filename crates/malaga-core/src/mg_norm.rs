//! Malagasy text normalization for TTS: turns written text into what a reader
//! says, spelled in Malagasy orthography (the only thing MMS-TTS can read).
//!
//! * numbers (`2024` → "efatra amby roapolo sy roa arivo"), decimals, ordinals
//!   (`faha-3`, `1er`), times (`14h30`), dates (`12/05/2024`), ranges, phone-like
//!   numbers (leading zero → digit by digit);
//! * units and currencies after a number (`5 km`, `2,5 kg`, `1 000 Ar`, `$5`, `15%`);
//! * symbols (`&`, `+`, `=`, `@`…), URLs and e-mail addresses;
//! * acronyms (`SMS` → "esy ema esy"), letters glued to numbers (`4G`);
//! * foreign words: a lexicon of common ones (`iphone` → "aifaona"), then
//!   respelling rules for any word that cannot be Malagasy (letters c q u w x,
//!   final consonant, impossible consonant cluster);
//! * brackets, colons, semicolons, dashes → `,` (a short pause downstream).
//!
//! Numbers follow the standard Malagasy order, smallest part first: the first
//! two parts are joined by `amby` (`ambin'ny` for 11–19), the rest by `sy`, and
//! 1 becomes `iraika` in front of `amby`/`ambin'ny`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use regex::{Captures, Regex};

const UNITS: [&str; 10] = ["aotra", "iray", "roa", "telo", "efatra", "dimy", "enina", "fito", "valo", "sivy"];
const TENS: [&str; 10] =
    ["", "folo", "roapolo", "telopolo", "efapolo", "dimampolo", "enimpolo", "fitopolo", "valopolo", "sivifolo"];
const HUNDREDS: [&str; 10] =
    ["", "zato", "roanjato", "telonjato", "efajato", "dimanjato", "eninjato", "fitonjato", "valonjato", "sivinjato"];
const MONTHS: [&str; 12] = [
    "janoary",
    "febroary",
    "martsa",
    "aprily",
    "mey",
    "jona",
    "jolay",
    "aogositra",
    "septambra",
    "oktobra",
    "novambra",
    "desambra",
];

/// Cardinal number in words.
pub fn number(n: u64) -> String {
    if n == 0 {
        return UNITS[0].into();
    }
    let low = n % 1_000_000;
    let digit = |p: u64| (low / p % 10) as usize;
    let (u, t, h) = (digit(1), digit(10), digit(100));
    let mut parts: Vec<String> = vec![];
    let mut leading_one = false;
    if t == 1 && u > 0 {
        parts.push(format!("{} ambin'ny folo", if u == 1 { "iraika" } else { UNITS[u] }));
    } else {
        if u > 0 {
            parts.push(UNITS[u].into());
            leading_one = u == 1;
        }
        if t > 0 {
            parts.push(TENS[t].into());
        }
    }
    if h > 0 {
        parts.push(HUNDREDS[h].into());
    }
    match digit(1000) {
        0 => {}
        1 => parts.push("arivo".into()),
        d => parts.push(format!("{} arivo", UNITS[d])),
    }
    for (p, name) in [(10_000, "alina"), (100_000, "hetsy")] {
        if digit(p) > 0 {
            parts.push(format!("{} {name}", UNITS[digit(p)]));
        }
    }
    let millions = n / 1_000_000 % 1000;
    if millions > 0 {
        parts.push(format!("{} tapitrisa", number(millions)));
    }
    let billions = n / 1_000_000_000;
    if billions > 0 {
        parts.push(format!("{} lavitrisa", number(billions)));
    }
    if parts.len() == 1 {
        return parts.remove(0);
    }
    if leading_one {
        parts[0] = "iraika".into();
    }
    let mut s = format!("{} amby {}", parts[0], parts[1]);
    for p in &parts[2..] {
        s.push_str(" sy ");
        s.push_str(p);
    }
    s
}

/// Digit by digit (phone numbers, codes, fractional parts with leading zeros).
pub fn digits(s: &str) -> String {
    s.chars().filter_map(|c| c.to_digit(10)).map(|d| UNITS[d as usize]).collect::<Vec<_>>().join(" ")
}

/// Integer string in words; digit by digit when it has a leading zero or is too long.
fn int_words(s: &str) -> String {
    if (s.len() > 1 && s.starts_with('0')) || s.len() > 15 {
        return digits(s);
    }
    s.parse::<u64>().map(number).unwrap_or_else(|_| digits(s))
}

/// `3,5` / `3.5` → "telo faingo dimy".
fn decimal_words(s: &str) -> String {
    match s.split_once([',', '.']) {
        Some((i, f)) => {
            let frac = if f.starts_with('0') { digits(f) } else { int_words(f) };
            format!("{} faingo {frac}", int_words(i))
        }
        None => int_words(s),
    }
}

/// Ordinal: 1 → "voalohany", 3 → "fahatelo", 4 → "fahefatra".
pub fn ordinal(n: u64) -> String {
    if n == 1 {
        return "voalohany".into();
    }
    let w = number(n);
    if w.starts_with(['a', 'e', 'i', 'o']) {
        format!("fah{w}")
    } else {
        format!("faha{w}")
    }
}

const LETTERS: [&str; 26] = [
    "a", "be", "se", "de", "e", "efa", "je", "asy", "i", "ji", "ka", "ela", "ema", "ena", "ô", "pe", "kio", "era",
    "esy", "te", "y", "ve", "dobilve", "iksy", "igreka", "zeda",
];

fn spell(word: &str) -> String {
    word.chars()
        .filter_map(|c| {
            let c = c.to_ascii_lowercase();
            c.is_ascii_lowercase().then(|| LETTERS[(c as u8 - b'a') as usize])
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Units / currencies after a number. Case matters (`m` metre vs `M`).
const UNIT_WORDS: &[(&str, &str)] = &[
    ("km/h", "kilometatra isan'ora"),
    ("m/s", "metatra isan-tsegondra"),
    ("km²", "kilometatra kare"),
    ("km2", "kilometatra kare"),
    ("m²", "metatra kare"),
    ("m2", "metatra kare"),
    ("m³", "metatra kiobe"),
    ("m3", "metatra kiobe"),
    ("kWh", "kilaoaty ora"),
    ("kW", "kilaoaty"),
    ("km", "kilometatra"),
    ("cm", "santimetatra"),
    ("mm", "milimetatra"),
    ("kg", "kilao"),
    ("mg", "miligrama"),
    ("ml", "mililitatra"),
    ("mL", "mililitatra"),
    ("cl", "santilitatra"),
    ("ha", "hektara"),
    ("Ar", "ariary"),
    ("ar", "ariary"),
    ("Fmg", "farantsa malagasy"),
    ("FMG", "farantsa malagasy"),
    ("Go", "jigaoktety"),
    ("GB", "jigaoktety"),
    ("Mo", "megaoktety"),
    ("MB", "megaoktety"),
    ("Ko", "kilaoktety"),
    ("KB", "kilaoktety"),
    ("To", "teraoktety"),
    ("TB", "teraoktety"),
    ("min", "minitra"),
    ("mn", "minitra"),
    ("ms", "milisegondra"),
    ("m", "metatra"),
    ("g", "grama"),
    ("t", "taonina"),
    ("l", "litatra"),
    ("L", "litatra"),
    ("h", "ora"),
    ("s", "segondra"),
    ("W", "oaty"),
    ("V", "volta"),
    ("°C", "degre selsiosy"),
    ("°", "degre"),
    ("%", "isan-jato"),
    ("‰", "isan'arivo"),
    ("€", "eoro"),
    ("$", "dolara"),
    ("£", "livatra"),
];

fn unit_word(u: &str) -> &'static str {
    UNIT_WORDS.iter().find(|(k, _)| *k == u).map(|(_, v)| *v).unwrap_or("")
}

/// Common foreign / brand / tech words, respelled the way Malagasy speakers say them.
const LEXICON: &[(&str, &str)] = &[
    ("iphone", "aifaona"),
    ("ipad", "aipady"),
    ("android", "andrôida"),
    ("ios", "ai ô esy"),
    ("google", "gogola"),
    ("gmail", "jimeila"),
    ("youtube", "iotioba"),
    ("facebook", "fesiboka"),
    ("messenger", "mesenjera"),
    ("whatsapp", "oatsapy"),
    ("instagram", "instagrama"),
    ("tiktok", "tiktôka"),
    ("twitter", "toitera"),
    ("telegram", "telegrama"),
    ("viber", "vaibera"),
    ("skype", "skaipy"),
    ("zoom", "zoma"),
    ("linkedin", "linkidina"),
    ("internet", "internety"),
    ("wifi", "oaifay"),
    ("wi-fi", "oaifay"),
    ("bluetooth", "blotosy"),
    ("email", "imeila"),
    ("e-mail", "imeila"),
    ("mail", "meila"),
    ("smartphone", "smartfaona"),
    ("phone", "faona"),
    ("laptop", "laptôpy"),
    ("ordinateur", "ôrdinatera"),
    ("computer", "kômpiotera"),
    ("tablette", "tableta"),
    ("clavier", "klavie"),
    ("écran", "ekrana"),
    ("logiciel", "lôjisiela"),
    ("application", "aplikasiôna"),
    ("app", "apy"),
    ("site", "sity"),
    ("web", "oeby"),
    ("online", "ônlaina"),
    ("offline", "ôflaina"),
    ("password", "pasoôrda"),
    ("login", "lôgina"),
    ("download", "daonlôda"),
    ("upload", "aplôda"),
    ("update", "apdeity"),
    ("chatgpt", "tsaty ji pi ti"),
    ("samsung", "samsonga"),
    ("apple", "apôly"),
    ("microsoft", "maikrôsôfty"),
    ("windows", "oindôza"),
    ("linux", "linoksa"),
    ("huawei", "hoaoe"),
    ("xiaomi", "siaômi"),
    ("nokia", "nôkia"),
    ("tecno", "teknô"),
    ("itel", "aitely"),
    ("airtel", "ertely"),
    ("orange", "ôranjy"),
    ("mvola", "emvôla"),
    ("netflix", "netfliksy"),
    ("spotify", "spôtifay"),
    ("amazon", "amazôna"),
    ("uber", "iobera"),
    ("covid", "kôvida"),
    ("virus", "viriosy"),
    ("vaccin", "vaksiny"),
    ("ok", "ôkey"),
    ("okay", "ôkey"),
    ("bye", "bay"),
    ("hello", "helô"),
    ("merci", "mersy"),
    ("bonjour", "bonjora"),
    ("salut", "saly"),
    ("président", "prezida"),
    ("gouvernement", "governemanta"),
    ("ministre", "minisitra"),
    ("football", "fotbôly"),
    ("match", "matsy"),
    ("club", "klioba"),
    ("bus", "bisy"),
    ("taxi", "taksy"),
    ("euro", "eoro"),
    ("euros", "eoro"),
    ("dollar", "dolara"),
    ("dollars", "dolara"),
    // Foreign names whose spelling looks Malagasy (no rule can catch them).
    ("tokyo", "tôkiô"),
    ("new", "nio"),
    ("york", "iôrka"),
    ("london", "londona"),
    ("toronto", "tôrôntô"),
    ("roma", "rôma"),
    ("moscou", "mosko"),
    ("dubai", "dobai"),
    ("mayotte", "maiôta"),
    ("comores", "kômôro"),
    ("maurice", "môrisy"),
    ("la", "la"),
    ("de", "de"),
    ("le", "le"),
];

/// Text produced by the expansions below is already Malagasy: it is wrapped in
/// these markers so the foreign-word pass leaves it alone.
const P0: char = '\u{E000}';
const P1: char = '\u{E001}';

fn protect(s: impl AsRef<str>) -> String {
    format!(" {P0}{}{P1} ", s.as_ref())
}

/// Letters MMS-mlg can read.
fn is_mg_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'y' | 'à' | 'ì' | 'ò' | 'ô' | 'ỳ')
}

/// Consonant clusters that occur in native Malagasy words.
const NATIVE_CLUSTERS: &[&str] =
    &["mb", "mp", "nd", "ndr", "nt", "ntr", "nts", "nj", "ng", "nk", "ts", "tr", "dr", "nz"];

/// True when a (lowercase) word cannot be native Malagasy.
pub fn looks_foreign(word: &str) -> bool {
    let native_letters = |c: char| is_mg_vowel(c) || "bdfghjklmnprstvz'-".contains(c);
    if !word.chars().all(native_letters) {
        return true;
    }
    // A part followed by an apostrophe (amin'ny) or a hyphen (isan-jato,
    // an-tanàna) may end in a consonant: elision / nasal linking.
    let parts: Vec<&str> = word.split(['-', '\'']).collect();
    let n = parts.len();
    let mut segments = parts.into_iter().enumerate().map(|(i, s)| (s, i + 1 < n));
    segments.any(|(seg, elided)| {
        let chars: Vec<char> = seg.chars().collect();
        if chars.is_empty() {
            return false;
        }
        if !elided && chars.len() > 1 && !is_mg_vowel(*chars.last().unwrap_or(&'a')) {
            return true;
        }
        seg.split(is_mg_vowel).any(|run| run.chars().count() > 1 && !NATIVE_CLUSTERS.contains(&run))
    })
}

/// Rewrites a foreign (mostly French / English) word in Malagasy orthography.
/// Approximate by nature: the lexicon (built-in or user) wins over these rules.
pub fn respell(word: &str) -> String {
    let c: Vec<char> = word
        .chars()
        .map(|c| match c {
            'é' => 'É', // pronounced, never a silent final e
            'è' | 'ê' | 'ë' | 'œ' | 'æ' => 'e',
            'â' | 'ä' => 'a',
            'î' | 'ï' => 'i',
            'ö' | 'ò' => 'o',
            'ù' | 'û' | 'ü' => 'u',
            'ÿ' => 'y',
            'ñ' => 'n',
            c => c,
        })
        .filter(|c| c.is_alphabetic() || *c == '\'' || *c == '-' || *c == 'ç')
        .collect();
    let vowel = |k: usize| c.get(k).is_some_and(|&ch| "aeiouyôàÉ".contains(ch));
    let rest = |i: usize, p: &str| c[i..].iter().collect::<String>().starts_with(p);
    let mut out = String::new();
    let mut i = 0;
    while i < c.len() {
        let next = c.get(i + 1).copied();
        let rule: Option<(usize, &str)> = [
            ("eau", "ô"),
            ("tion", "siôn"),
            ("ou", "o"),
            ("oo", "o"),
            ("oi", "oa"),
            ("oy", "oay"),
            ("au", "ô"),
            ("ai", "e"),
            ("ei", "e"),
            ("eu", "e"),
            ("ee", "i"),
            ("ea", "i"),
            ("ph", "f"),
            ("th", "t"),
            ("sch", "s"),
            ("sh", "s"),
            ("ch", "s"),
            ("ck", "k"),
            ("qu", "k"),
            ("gn", "ny"),
        ]
        .iter()
        .find(|(p, _)| rest(i, p))
        .map(|(p, r)| (p.chars().count(), *r));
        if let Some((n, r)) = rule {
            out.push_str(r);
            i += n;
            continue;
        }
        let ch = c[i];
        let soft = matches!(next, Some('e' | 'i' | 'y'));
        let piece: String = match ch {
            'c' if soft => "s".into(),
            'c' | 'q' => "k".into(),
            'ç' => "s".into(),
            'x' => "ks".into(),
            'w' => "o".into(),
            'u' => "i".into(),
            'o' => "ô".into(),
            'É' => "e".into(),
            'g' if soft => "j".into(),
            'h' => String::new(),
            's' if i > 0 && vowel(i - 1) && vowel(i + 1) => "z".into(),
            // Silent French final e after a consonant (france, apple).
            'e' if i + 1 == c.len() && i >= 2 && !vowel(i - 1) => String::new(),
            ch => ch.to_string(),
        };
        // Doubled consonants collapse.
        if piece.chars().count() == 1 && !is_mg_vowel(ch) && out.ends_with(&piece) {
            i += 1;
            continue;
        }
        out.push_str(&piece);
        i += 1;
    }
    // Malagasy words end in a vowel: add the usual loanword ending.
    if let Some(last) = out.chars().last() {
        if !is_mg_vowel(last) && last.is_alphabetic() {
            out.push(if matches!(last, 'n' | 'm' | 'r') { 'a' } else { 'y' });
        }
    }
    out
}

struct Patterns {
    url: Regex,
    phone: Regex,
    email: Regex,
    thousands: Regex,
    time: Regex,
    hour: Regex,
    date: Regex,
    faha: Regex,
    french_ord: Regex,
    range: Regex,
    currency_prefix: Regex,
    unit_symbol: Regex,
    unit_word: Regex,
    num_letters: Regex,
    letter_num: Regex,
    word_num: Regex,
    negative: Regex,
    decimal: Regex,
    integer: Regex,
    word: Regex,
    spaces: Regex,
    commas: Regex,
}

fn patterns() -> &'static Patterns {
    static P: OnceLock<Patterns> = OnceLock::new();
    P.get_or_init(|| {
        let r = |s: &str| Regex::new(s).unwrap_or_else(|e| panic!("bad regex {s}: {e}"));
        let mut words: Vec<&str> =
            UNIT_WORDS.iter().map(|(k, _)| *k).filter(|k| k.chars().all(|c| c.is_ascii_alphanumeric() || c == '/')).collect();
        words.sort_by_key(|w| std::cmp::Reverse(w.len()));
        let words: Vec<String> = words.iter().map(|w| regex::escape(w)).collect();
        Patterns {
            url: r(r"(?i)\b(?:https?://)?(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|org|net|mg|fr|io|edu|gov|info)\b(?:/\S*)?"),
            phone: r(r"(?:\+\d{2,3}[ .]?)?\b0\d{2}(?:[ .-]\d{2,3}){2,4}\b"),
            email: r(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"),
            thousands: r(r"\b(\d{1,3})((?:[ .]\d{3})+)\b"),
            time: r(r"\b(\d{1,2})\s?[hH:]\s?(\d{2})\b"),
            hour: r(r"\b(\d{1,2})\s?[hH]\b"),
            date: r(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4}|\d{2})\b"),
            faha: r(r"(?i)\bfaha-?(\d+)\b"),
            french_ord: r(r"\b(\d+)(?:er|ère|re|ème|eme|e)\b"),
            range: r(r"\b([1-9]\d*(?:[.,]\d+)?)\s?[-–]\s?([1-9]\d*(?:[.,]\d+)?)\b"),
            currency_prefix: r(r"([$€£])\s?(\d+(?:[.,]\d+)?)"),
            unit_symbol: r(r"(\d+(?:[.,]\d+)?)\s?(°C|km²|m²|m³|%|‰|°|€|\$|£)"),
            unit_word: r(&format!(r"(\d+(?:[.,]\d+)?)\s?({})\b", words.join("|"))),
            num_letters: r(r"\b(\d+)([A-Z]{1,2})\b"),
            letter_num: r(r"\b([A-Z]{1,2})(\d+)\b"),
            word_num: r(r"(\p{L})-(\d)"),
            negative: r(r"(^|\s)-(\d)"),
            decimal: r(r"\b(\d+[,.]\d+)\b"),
            integer: r(r"\d+"),
            word: r(r"[\p{L}][\p{L}'-]*"),
            spaces: r(r"[ \t]+"),
            commas: r(r"\s*,[\s,]*"),
        }
    })
}

/// Text normalizer with the built-in lexicon plus optional user entries.
#[derive(Debug, Clone)]
pub struct MgNormalizer {
    lexicon: HashMap<String, String>,
}

impl Default for MgNormalizer {
    fn default() -> Self {
        Self { lexicon: LEXICON.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect() }
    }
}

impl MgNormalizer {
    /// Adds `word<TAB>respelling` lines (`#` comments allowed); they override
    /// the built-in entries. Keys are matched case-insensitively.
    pub fn load_lexicon(&mut self, path: &Path) -> Result<usize> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading lexicon {}", path.display()))?;
        let mut n = 0;
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
            let (k, v) = line
                .split_once('\t')
                .or_else(|| line.split_once('='))
                .with_context(|| format!("lexicon line without TAB or '=': {line}"))?;
            self.lexicon.insert(k.trim().to_lowercase(), v.trim().to_string());
            n += 1;
        }
        Ok(n)
    }

    fn word(&self, w: &str) -> String {
        let lower = w.to_lowercase();
        if let Some(v) = self.lexicon.get(&lower) {
            return v.clone();
        }
        let letters = w.chars().filter(|c| c.is_alphabetic()).count();
        let upper = w.chars().filter(|c| c.is_uppercase()).count();
        if letters >= 2 && upper == letters {
            let has_vowel = w.chars().any(|c| "AEIOUY".contains(c));
            if letters <= 3 || !has_vowel {
                return spell(w);
            }
        }
        if looks_foreign(&lower) {
            // Hyphenated words: respell only the foreign parts (a non-final part
            // may end in a consonant, as in isan-jato).
            let parts: Vec<&str> = lower.split('-').collect();
            let last = parts.len() - 1;
            return parts
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let foreign = if i < last { looks_foreign(&format!("{p}-")) } else { looks_foreign(p) };
                    if foreign {
                        respell(p)
                    } else {
                        p.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("-");
        }
        lower
    }

    /// Written text → spoken Malagasy text, keeping `.`, `!`, `?` (sentence
    /// ends) and `,` (short pauses).
    pub fn normalize(&self, text: &str) -> String {
        use unicode_normalization::UnicodeNormalization;
        let p = patterns();
        let mut s: String = text
            .nfc()
            .map(|c| match c {
                '\u{a0}' | '\u{202f}' | '\u{2009}' => ' ',
                '’' | '‘' | '`' | 'ʼ' => '\'',
                c => c,
            })
            .collect();
        let rep = |s: &str, re: &Regex, f: &dyn Fn(&Captures) -> String| {
            re.replace_all(s, |c: &Captures| {
                let out = f(c);
                // A refused match (e.g. 25:99 is not a time) is returned unchanged.
                if out == c[0] {
                    out
                } else {
                    protect(out)
                }
            })
            .into_owned()
        };

        // Addresses: every piece read on its own, dots as "teboka", short
        // vowel-less pieces (mg, fr, www) spelled.
        let address = |m: &str| -> String {
            let m = m.to_lowercase();
            let m = m.trim_start_matches("https://").trim_start_matches("http://").trim_start_matches("www.");
            let mut out = vec![];
            for (i, part) in m.split('.').enumerate() {
                if i > 0 {
                    out.push("teboka".to_string());
                }
                for (j, piece) in part.split('@').enumerate() {
                    if j > 0 {
                        out.push("arobasy".to_string());
                    }
                    for w in piece.split(['/', '-', '_', '?', '=', '&']).filter(|w| !w.is_empty()) {
                        let vowelless = !w.chars().any(|c| "aeiouy".contains(c));
                        out.push(if vowelless && w.len() <= 4 { spell(w) } else { self.word(w) });
                    }
                }
            }
            out.join(" ")
        };
        s = rep(&s, &p.email, &|c| address(&c[0]));
        s = rep(&s, &p.url, &|c| address(&c[0]));
        // Phone numbers: one group at a time, with a short pause between groups.
        s = rep(&s, &p.phone, &|c| {
            let groups: Vec<String> = c[0]
                .split([' ', '.', '-'])
                .filter(|g| !g.is_empty())
                .map(|g| match g.strip_prefix('+') {
                    Some(cc) => format!("miampy {}", int_words(cc)),
                    None => int_words(g),
                })
                .collect();
            format!(" {}, ", groups.join(", "))
        });
        s = rep(&s, &p.date, &|c| {
            let (d, m): (u64, usize) = (c[1].parse().unwrap_or(0), c[2].parse().unwrap_or(0));
            if !(1..=31).contains(&d) || !(1..=12).contains(&m) {
                return c[0].to_string();
            }
            let day = if d == 1 { "voalohany".to_string() } else { number(d) };
            format!("{day} {} {}", MONTHS[m - 1], int_words(&c[3]))
        });
        // Digits only (no words yet): not protected, so units can still follow.
        s = p
            .thousands
            .replace_all(&s, |c: &Captures| format!("{}{}", &c[1], c[2].replace([' ', '.'], "")))
            .into_owned();
        s = rep(&s, &p.time, &|c| {
            let (h, m): (u64, u64) = (c[1].parse().unwrap_or(0), c[2].parse().unwrap_or(0));
            if h > 24 || m > 59 {
                return c[0].to_string();
            }
            match m {
                0 => format!("{} ora", number(h)),
                m => format!("{} ora sy {} minitra", number(h), number(m)),
            }
        });
        s = rep(&s, &p.hour, &|c| format!("{} ora", int_words(&c[1])));
        s = rep(&s, &p.faha, &|c| c[1].parse().map(ordinal).unwrap_or_default());
        s = rep(&s, &p.french_ord, &|c| c[1].parse().map(ordinal).unwrap_or_default());
        s = rep(&s, &p.range, &|c| format!("{} ka hatramin'ny {}", decimal_words(&c[1]), decimal_words(&c[2])));
        s = rep(&s, &p.currency_prefix, &|c| format!("{} {}", decimal_words(&c[2]), unit_word(&c[1])));
        s = rep(&s, &p.unit_symbol, &|c| format!("{} {}", decimal_words(&c[1]), unit_word(&c[2])));
        s = rep(&s, &p.unit_word, &|c| format!("{} {}", decimal_words(&c[1]), unit_word(&c[2])));
        s = rep(&s, &p.num_letters, &|c| format!("{} {}", int_words(&c[1]), spell(&c[2])));
        s = rep(&s, &p.letter_num, &|c| format!("{} {}", spell(&c[1]), int_words(&c[2])));
        s = rep(&s, &p.word_num, &|c| format!("{} {}", &c[1], &c[2]));
        s = rep(&s, &p.negative, &|c| format!("{}latsaka {}", &c[1], &c[2]));
        s = rep(&s, &p.decimal, &|c| decimal_words(&c[1]));
        s = rep(&s, &p.integer, &|c| format!(" {} ", int_words(&c[0])));

        // Symbols and punctuation.
        let mut t = String::with_capacity(s.len());
        let chars: Vec<char> = s.chars().collect();
        for (i, &ch) in chars.iter().enumerate() {
            let prev_letter = i > 0 && chars[i - 1].is_alphanumeric();
            let next_letter = chars.get(i + 1).is_some_and(|c| c.is_alphanumeric());
            match ch {
                '&' => t.push_str(" sy "),
                '+' => t.push_str(" miampy "),
                '=' => t.push_str(" mitovy amin'ny "),
                '@' => t.push_str(" arobasy "),
                '#' => t.push_str(" diezy "),
                '%' => t.push_str(" isan-jato "),
                '€' => t.push_str(" eoro "),
                '$' => t.push_str(" dolara "),
                '£' => t.push_str(" livatra "),
                '°' => t.push_str(" degre "),
                '§' => t.push_str(" fizarana "),
                '×' => t.push_str(" ampitomboina "),
                '÷' => t.push_str(" zaraina "),
                '<' => t.push_str(" latsaky ny "),
                '>' => t.push_str(" mihoatra ny "),
                '~' | '≈' => t.push_str(" eo ho eo "),
                '/' if prev_letter && next_letter => t.push_str(" na "),
                '(' | ')' | '[' | ']' | '{' | '}' | ':' | ';' | '—' | '–' | '|' | '→' | '•' => t.push_str(", "),
                '-' if !(prev_letter && next_letter) => t.push_str(", "),
                '"' | '«' | '»' | '“' | '”' | '*' | '_' | '/' | '\\' | '^' => t.push(' '),
                '\'' if !(prev_letter && next_letter) => t.push(' '),
                '\n' => t.push_str(".\n"),
                c => t.push(c),
            }
        }

        // Words: lexicon, acronyms, foreign respelling.
        let mut w = String::with_capacity(t.len());
        for (i, chunk) in t.split([P0, P1]).enumerate() {
            // Odd chunks sit between P0 and P1: generated, already Malagasy.
            if i % 2 == 1 {
                w.push_str(chunk);
            } else {
                w.push_str(&p.word.replace_all(chunk, |c: &Captures| self.word(&c[0])));
            }
        }
        let t = w;
        let t = p.spaces.replace_all(&t, " ");
        let t = p.commas.replace_all(&t, ", ");
        // Drop pauses glued to sentence ends or text edges.
        let t = t.replace(", .", ".").replace(" .", ".").replace(" ?", "?").replace(" !", "!");
        t.trim_matches(|c: char| c == ',' || c.is_whitespace()).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers() {
        assert_eq!(number(0), "aotra");
        assert_eq!(number(1), "iray");
        assert_eq!(number(10), "folo");
        assert_eq!(number(11), "iraika ambin'ny folo");
        assert_eq!(number(15), "dimy ambin'ny folo");
        assert_eq!(number(20), "roapolo");
        assert_eq!(number(21), "iraika amby roapolo");
        assert_eq!(number(45), "dimy amby efapolo");
        assert_eq!(number(100), "zato");
        assert_eq!(number(101), "iraika amby zato");
        assert_eq!(number(125), "dimy amby roapolo sy zato");
        assert_eq!(number(115), "dimy ambin'ny folo amby zato");
        assert_eq!(number(1000), "arivo");
        assert_eq!(number(1990), "sivifolo amby sivinjato sy arivo");
        assert_eq!(number(2024), "efatra amby roapolo sy roa arivo");
        assert_eq!(number(25_000), "dimy arivo amby roa alina");
        assert_eq!(number(1_000_000), "iray tapitrisa");
        assert_eq!(number(3_500_000), "dimy hetsy amby telo tapitrisa");
        assert_eq!(ordinal(1), "voalohany");
        assert_eq!(ordinal(3), "fahatelo");
        assert_eq!(ordinal(4), "fahefatra");
    }

    #[test]
    fn foreign_detection() {
        for w in [
            "madagasikara",
            "antananarivo",
            "mpianatra",
            "fianarantsoa",
            "amin'ny",
            "isa-maraina",
            "isan-jato",
            "an-tanàna",
            "tsy",
            "ny",
        ] {
            assert!(!looks_foreign(w), "{w}");
        }
        for w in ["iphone", "android", "google", "internet", "paris", "club", "michel"] {
            assert!(looks_foreign(w), "{w}");
        }
        assert_eq!(respell("paris"), "parisy");
        assert_eq!(respell("france"), "fransy");
        assert_eq!(respell("ordinateur"), "ôrdinatera");
        assert_eq!(respell("michel"), "misely");
        assert_eq!(respell("santé"), "sante");
    }

    #[test]
    fn sentences() {
        let n = MgNormalizer::default();
        assert_eq!(
            n.normalize("Nividy iPhone 15 sy Android aho."),
            "nividy aifaona dimy ambin'ny folo sy andrôida aho."
        );
        assert_eq!(n.normalize("Lavitra 5 km (eo ho eo)."), "lavitra dimy kilometatra, eo ho eo.");
        assert_eq!(
            n.normalize("Mitentina 2,5 kg sy 1 500 Ar"),
            "mitentina roa faingo dimy kilao sy dimanjato amby arivo ariary"
        );
        assert_eq!(
            n.normalize("Amin'ny 14h30 ny SMS"),
            "amin'ny efatra ambin'ny folo ora sy telopolo minitra ny esy ema esy"
        );
        assert_eq!(n.normalize("15% amin'ny mponina"), "dimy ambin'ny folo isan-jato amin'ny mponina");
        assert_eq!(n.normalize("Ny maripana dia 28°C"), "ny maripana dia valo amby roapolo degre selsiosy");
        assert_eq!(n.normalize("Galaxy S24"), "galaksy esy efatra amby roapolo");
        assert_eq!(n.normalize("info@malaga.mg"), "infô arobasy malaga teboka ema je");
        assert_eq!(
            n.normalize("Antsoy ny 034 12 345 67"),
            "antsoy ny aotra telo efatra, roa ambin'ny folo, dimy amby efapolo sy telonjato, fito amby enimpolo"
        );
    }
}
