//! Minimal JSON-lines helpers for `zerolm live --jsonl` (std only, no serde).
//! Input line: plain text, or an object with a string field `"text"`.
//! Output line: one JSON object per turn.

pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn hex4(chars: &mut std::str::Chars<'_>) -> Option<u32> {
    let mut v = 0u32;
    for _ in 0..4 {
        v = v * 16 + chars.next()?.to_digit(16)?;
    }
    Some(v)
}

/// Parse a JSON string body starting right after the opening quote.
fn parse_string(rest: &str) -> Option<String> {
    let mut out = String::new();
    let mut it = rest.chars();
    loop {
        match it.next()? {
            '"' => return Some(out),
            '\\' => match it.next()? {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                '/' => out.push('/'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'b' => out.push('\u{8}'),
                'f' => out.push('\u{c}'),
                'u' => {
                    let hi = hex4(&mut it)?;
                    let cp = if (0xD800..0xDC00).contains(&hi) {
                        let mut peek = it.clone();
                        if peek.next() == Some('\\') && peek.next() == Some('u') {
                            match hex4(&mut peek) {
                                Some(lo) if (0xDC00..0xE000).contains(&lo) => {
                                    it = peek;
                                    0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                                }
                                _ => 0xFFFD,
                            }
                        } else {
                            0xFFFD
                        }
                    } else {
                        hi
                    };
                    out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                }
                _ => return None,
            },
            c => out.push(c),
        }
    }
}

/// Value of the `"text"` key of a JSON object line, if present as a string.
pub fn extract_text(line: &str) -> Option<String> {
    let mut from = 0;
    while let Some(i) = line[from..].find("\"text\"") {
        let after = from + i + "\"text\"".len();
        let rest = line[after..].trim_start();
        if let Some(rest) = rest.strip_prefix(':') {
            if let Some(body) = rest.trim_start().strip_prefix('"') {
                return parse_string(body);
            }
        }
        from = after;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_specials() {
        assert_eq!(escape("a\"b\\c\nd\u{1}"), "a\\\"b\\\\c\\nd\\u0001");
    }

    #[test]
    fn extracts_text_with_escapes_and_unicode() {
        assert_eq!(extract_text(r#"{"text":"hi \"you\"\n"}"#).unwrap(), "hi \"you\"\n");
        assert_eq!(extract_text(r#"{ "text" : "café 😀" }"#).unwrap(), "café 😀");
    }

    #[test]
    fn skips_text_used_as_a_value_and_rejects_non_strings() {
        assert_eq!(extract_text(r#"{"kind":"text","text":"real"}"#).unwrap(), "real");
        assert!(extract_text(r#"{"text": 5}"#).is_none());
        assert!(extract_text(r#"{"other":"x"}"#).is_none());
        assert!(extract_text(r#"{"text":"unterminated"#).is_none());
    }
}
