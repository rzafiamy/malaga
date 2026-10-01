//! Splits a document into sentences while remembering the separators, so the
//! translation can be stitched back with the original layout (newlines,
//! paragraphs). NLLB was trained on single sentences: feeding it one sentence
//! at a time is both better quality and lets the sentences be batched.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Segment<'a> {
    Text(&'a str),
    Sep(&'a str),
}

const ABBREVIATIONS: &[&str] = &[
    "M", "Mme", "Mlle", "Dr", "Pr", "St", "Ste", "Mr", "Mrs", "Ms", "Prof", "etc", "cf", "p", "n", "vol", "ex", "e.g",
    "i.e", "vs", "Jr", "Sr", "No", "art",
];

fn is_abbreviation(before: &str) -> bool {
    let word = before.rsplit(|c: char| c.is_whitespace() || c == '(').next().unwrap_or_default();
    let word = word.trim_end_matches('.');
    ABBREVIATIONS.iter().any(|a| a.eq_ignore_ascii_case(word))
        || (word.chars().count() == 1 && word.chars().all(|c| c.is_uppercase()))
}

pub fn segment(text: &str) -> Vec<Segment<'_>> {
    let mut out = Vec::new();
    for (i, line) in text.split('\n').enumerate() {
        if i > 0 {
            out.push(Segment::Sep("\n"));
        }
        split_line(line, &mut out);
    }
    out
}

fn split_line<'a>(line: &'a str, out: &mut Vec<Segment<'a>>) {
    let body = line.trim();
    if body.is_empty() {
        if !line.is_empty() {
            out.push(Segment::Sep(line));
        }
        return;
    }
    let trimmed_start = line.len() - line.trim_start().len();
    if trimmed_start > 0 {
        out.push(Segment::Sep(&line[..trimmed_start]));
    }
    let chars: Vec<(usize, char)> = body.char_indices().collect();
    let mut start = 0;
    let mut i = 0;
    while i < chars.len() {
        let (pos, c) = chars[i];
        if matches!(c, '.' | '!' | '?' | '…' | '。') {
            // Include trailing punctuation / closing quotes in the sentence.
            let skip_ws = |mut k: usize| {
                while k < chars.len() && chars[k].1.is_whitespace() {
                    k += 1;
                }
                k
            };
            let mut j = i + 1;
            loop {
                if j < chars.len() && matches!(chars[j].1, '.' | '!' | '?' | '"' | '»' | '”' | '\'' | ')' | '’') {
                    j += 1;
                } else if skip_ws(j) < chars.len() && matches!(chars[skip_ws(j)].1, '»' | '”') {
                    // French spacing: « oui. »
                    j = skip_ws(j) + 1;
                } else {
                    break;
                }
            }
            let ws_end = skip_ws(j);
            let followed_by_start = ws_end > j
                && ws_end < chars.len()
                && (chars[ws_end].1.is_uppercase()
                    || chars[ws_end].1.is_numeric()
                    || matches!(chars[ws_end].1, '"' | '«' | '“' | '(' | '-' | '—'));
            if followed_by_start && !(c == '.' && is_abbreviation(&body[start..pos])) {
                let end = chars.get(j).map(|x| x.0).unwrap_or(body.len());
                let next = chars[ws_end].0;
                out.push(Segment::Text(&body[start..end]));
                out.push(Segment::Sep(&body[end..next]));
                start = next;
                i = ws_end;
                continue;
            }
            i = j;
            continue;
        }
        i += 1;
    }
    out.push(Segment::Text(&body[start..]));
    let tail = &line[trimmed_start + body.len()..];
    if !tail.is_empty() {
        out.push(Segment::Sep(tail));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(s: &str) -> Vec<&str> {
        segment(s)
            .into_iter()
            .filter_map(|s| match s {
                Segment::Text(t) => Some(t),
                _ => None,
            })
            .collect()
    }

    fn rebuild(s: &str) -> String {
        segment(s)
            .into_iter()
            .map(|s| match s {
                Segment::Text(t) | Segment::Sep(t) => t,
            })
            .collect()
    }

    /// covers: REQ-INF-004
    #[test]
    fn sentences() {
        assert_eq!(
            texts("Bonjour M. Dupont. Comment allez-vous ? Très bien !"),
            vec!["Bonjour M. Dupont.", "Comment allez-vous ?", "Très bien !"]
        );
        assert_eq!(texts("Hello world"), vec!["Hello world"]);
        assert_eq!(texts("Version 2.5 is out. Great."), vec!["Version 2.5 is out.", "Great."]);
        assert_eq!(texts("Il a dit « oui. » Puis il est parti."), vec!["Il a dit « oui. »", "Puis il est parti."]);
    }

    /// covers: REQ-INF-004
    #[test]
    fn layout_is_preserved() {
        for s in ["A. B.\n\n  C!  \nD", "", "\n", "x\n", "  lead", "Un. Deux.   Trois."] {
            assert_eq!(rebuild(s), s);
        }
    }
}
