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
    parse_string_chars(&mut rest.chars(), false)
}

/// Like `parse_string`, advancing `it` past the closing quote. In `strict`
/// mode raw control characters (below U+0020) are rejected, as RFC 8259 requires.
fn parse_string_chars(it: &mut std::str::Chars<'_>, strict: bool) -> Option<String> {
    let mut out = String::new();
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
                    let hi = hex4(it)?;
                    let cp = if (0xD800..0xDC00).contains(&hi) {
                        let mut peek = it.clone();
                        if peek.next() == Some('\\') && peek.next() == Some('u') {
                            match hex4(&mut peek) {
                                Some(lo) if (0xDC00..0xE000).contains(&lo) => {
                                    *it = peek;
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
            c if strict && (c as u32) < 0x20 => return None,
            c => out.push(c),
        }
    }
}

/// A parsed JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Str(String),
    Num(f64),
    Bool(bool),
    Null,
    Arr(Vec<Json>),
    Obj(std::collections::HashMap<String, Json>),
}

impl Json {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.get(key),
            _ => None,
        }
    }
}

const MAX_DEPTH: usize = 8;

fn skip_ws(it: &mut std::str::Chars<'_>) {
    while matches!(it.clone().next(), Some(' ' | '\t' | '\n' | '\r')) {
        it.next();
    }
}

fn parse_value(it: &mut std::str::Chars<'_>, depth: usize) -> Option<Json> {
    if depth > MAX_DEPTH {
        return None;
    }
    skip_ws(it);
    match it.clone().next()? {
        '"' => {
            it.next();
            parse_string_chars(it, true).map(Json::Str)
        }
        '{' => parse_members(it, depth + 1).map(Json::Obj),
        '[' => {
            it.next();
            skip_ws(it);
            let mut items = Vec::new();
            if it.clone().next()? == ']' {
                it.next();
                return Some(Json::Arr(items));
            }
            loop {
                items.push(parse_value(it, depth + 1)?);
                skip_ws(it);
                match it.next()? {
                    ',' => continue,
                    ']' => return Some(Json::Arr(items)),
                    _ => return None,
                }
            }
        }
        _ => {
            let mut tok = String::new();
            while let Some(c) = it.clone().next() {
                if matches!(c, ',' | '}' | ']' | ' ' | '\t' | '\n' | '\r') {
                    break;
                }
                tok.push(c);
                it.next();
            }
            match tok.as_str() {
                "true" => Some(Json::Bool(true)),
                "false" => Some(Json::Bool(false)),
                "null" => Some(Json::Null),
                t if !t.is_empty()
                    && t.chars().all(|c| c.is_ascii_digit() || matches!(c, '+' | '-' | '.' | 'e' | 'E')) =>
                {
                    t.parse::<f64>().ok().filter(|n| n.is_finite()).map(Json::Num)
                }
                _ => None,
            }
        }
    }
}

fn parse_members(
    it: &mut std::str::Chars<'_>,
    depth: usize,
) -> Option<std::collections::HashMap<String, Json>> {
    let mut map = std::collections::HashMap::new();
    skip_ws(it);
    if it.next()? != '{' {
        return None;
    }
    skip_ws(it);
    if it.clone().next()? == '}' {
        it.next();
        return Some(map);
    }
    loop {
        skip_ws(it);
        if it.next()? != '"' {
            return None;
        }
        let key = parse_string_chars(it, true)?;
        skip_ws(it);
        if it.next()? != ':' {
            return None;
        }
        let value = parse_value(it, depth)?;
        map.insert(key, value); // duplicate keys: last wins
        skip_ws(it);
        match it.next()? {
            ',' => continue,
            '}' => return Some(map),
            _ => return None,
        }
    }
}

/// Parse a complete JSON document (any value) with no trailing data and nesting of at
/// most 8 levels. Strings are checked strictly (no raw control characters).
pub fn parse(body: &str) -> Option<Json> {
    let mut it = body.chars();
    let v = parse_value(&mut it, 0)?;
    skip_ws(&mut it);
    it.next().is_none().then_some(v)
}

/// Parse a JSON document that must be a single object, and return its string-valued
/// members. Other member types are validated but ignored. Returns `None` for anything
/// that is not well-formed JSON (including trailing data and nesting deeper than 8).
pub fn parse_object(body: &str) -> Option<std::collections::HashMap<String, String>> {
    let mut it = body.chars();
    let members = parse_members(&mut it, 0)?;
    skip_ws(&mut it);
    if it.next().is_some() {
        return None;
    }
    Some(
        members
            .into_iter()
            .filter_map(|(k, v)| match v {
                Json::Str(s) => Some((k, s)),
                _ => None,
            })
            .collect(),
    )
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
    fn parse_object_accepts_valid_json_and_keeps_string_members() {
        let m = parse_object(r#" {"text":"hi\n","n":5,"ok":true,"x":null,"o":{"a":[1,{"b":"c"}]},"l":[]} "#).unwrap();
        assert_eq!(m.get("text").map(String::as_str), Some("hi\n"));
        assert_eq!(m.len(), 1, "non-string members are ignored: {m:?}");
        assert_eq!(parse_object("{}").unwrap().len(), 0);
        assert_eq!(parse_object(r#"{"a":"1","a":"2"}"#).unwrap()["a"], "2");
    }

    #[test]
    fn parse_builds_a_full_tree() {
        let v = parse(r#"{"model":"m","n":2.5,"stream":true,"x":null,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#).unwrap();
        assert_eq!(v.get("model").and_then(Json::as_str), Some("m"));
        assert_eq!(v.get("n").and_then(Json::as_f64), Some(2.5));
        assert_eq!(v.get("stream").and_then(Json::as_bool), Some(true));
        assert_eq!(v.get("x"), Some(&Json::Null));
        let msg = &v.get("messages").and_then(Json::as_array).unwrap()[0];
        assert_eq!(msg.get("role").and_then(Json::as_str), Some("user"));
        let part = &msg.get("content").and_then(Json::as_array).unwrap()[0];
        assert_eq!(part.get("text").and_then(Json::as_str), Some("hi"));
        assert_eq!(parse("[1, 2 ,3]").unwrap().as_array().unwrap().len(), 3);
        assert!(parse("1e999").is_none(), "non-finite numbers are rejected");
        assert!(parse("[1,]").is_none() && parse("{} {}").is_none() && parse("").is_none());
    }

    #[test]
    fn json_unicode_escapes_decode_including_surrogate_pairs() {
        // é = e-acute; 😀 = U+1F600 as a surrogate pair.
        let m = parse_object(r#"{"t":"café 😀 A"}"#).unwrap();
        assert_eq!(m["t"], "caf\u{e9} \u{1F600} A");
        // A lone surrogate becomes U+FFFD instead of being rejected or panicking.
        assert_eq!(parse_object(r#"{"t":"\ud83d x"}"#).unwrap()["t"], "\u{FFFD} x");
        assert!(parse_object(r#"{"t":"\u12"}"#).is_none(), "truncated escape");
        assert!(parse_object(r#"{"t":"\u12zz"}"#).is_none(), "non-hex escape");
    }

    #[test]
    fn parse_object_rejects_malformed_input() {
        for bad in [
            "", "text", "[]", "\"x\"", "{", "{\"a\"}", "{\"a\":}", "{\"a\":\"x\",}", "{\"a\":\"x\"} x",
            "{\"a\":\"x\"}{}", "{'a':'x'}", "{\"a\":tru}", "{\"a\":01x}", "{\"a\":\"unterminated}",
            "{\"a\":\"bad\\q\"}", "{\"a\":\"raw\ttab\"}", "{\"a\":\"nul\u{0}\"}",
        ] {
            assert!(parse_object(bad).is_none(), "should reject {bad:?}");
        }
        let deep = format!("{}{}", "[".repeat(20), "]".repeat(20));
        assert!(parse_object(&format!("{{\"a\":{deep}}}")).is_none(), "depth limit");
    }

    #[test]
    fn skips_text_used_as_a_value_and_rejects_non_strings() {
        assert_eq!(extract_text(r#"{"kind":"text","text":"real"}"#).unwrap(), "real");
        assert!(extract_text(r#"{"text": 5}"#).is_none());
        assert!(extract_text(r#"{"other":"x"}"#).is_none());
        assert!(extract_text(r#"{"text":"unterminated"#).is_none());
    }
}
