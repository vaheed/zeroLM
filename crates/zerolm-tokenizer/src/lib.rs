//! IDs 0..=255 are raw bytes. 256.. are control IDs. There is no learned or
//! imported vocabulary.

pub const BYTE_COUNT: usize = 256;
pub const BOS: u16 = 256;
pub const EOS: u16 = 257;
pub const PAD: u16 = 258;
pub const VOCAB_SIZE: usize = 259;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    pub id: u16,
    /// Byte offset in the source text (control tokens: offset of insertion).
    pub offset: usize,
}

/// Encode text as one token per UTF-8 byte.
pub fn encode(text: &str) -> Vec<Token> {
    text.bytes().enumerate().map(|(offset, b)| Token { id: b as u16, offset }).collect()
}

/// Encode with BOS/EOS markers.
pub fn encode_framed(text: &str) -> Vec<Token> {
    let mut v = vec![Token { id: BOS, offset: 0 }];
    v.extend(encode(text));
    v.push(Token { id: EOS, offset: text.len() });
    v
}

/// Raw bytes of the byte tokens; control tokens are dropped.
pub fn decode_bytes(tokens: &[Token]) -> Vec<u8> {
    tokens.iter().filter(|t| (t.id as usize) < BYTE_COUNT).map(|t| t.id as u8).collect()
}

/// Lossy UTF-8 decode (invalid sequences become U+FFFD).
pub fn decode(tokens: &[Token]) -> String {
    String::from_utf8_lossy(&decode_bytes(tokens)).into_owned()
}

/// Any byte sequence is representable, including invalid UTF-8.
pub fn encode_bytes(bytes: &[u8]) -> Vec<Token> {
    bytes.iter().enumerate().map(|(offset, &b)| Token { id: b as u16, offset }).collect()
}

/// Byte ranges (start, end) of invalid UTF-8 sequences. Empty == valid.
/// Invalid input is still tokenizable; this only reports where it is malformed.
pub fn invalid_utf8_spans(bytes: &[u8]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut pos = 0;
    while pos < bytes.len() {
        match std::str::from_utf8(&bytes[pos..]) {
            Ok(_) => break,
            Err(e) => {
                let start = pos + e.valid_up_to();
                // error_len() == None means a truncated sequence at the end.
                let end = start + e.error_len().unwrap_or(bytes.len() - start);
                spans.push((start, end));
                pos = end;
            }
        }
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_unicode_and_preserves_offsets() {
        let s = "héllo, 世界 \u{1F600}\n\0";
        let t = encode(s);
        assert_eq!(t.len(), s.len());
        assert!(t.iter().enumerate().all(|(i, t)| t.offset == i));
        assert_eq!(decode(&t), s);
    }

    #[test]
    fn arbitrary_bytes_are_representable() {
        let all: Vec<u8> = (0..=255).collect();
        assert_eq!(decode_bytes(&encode_bytes(&all)), all);
    }

    #[test]
    fn framing_adds_control_ids_that_decode_drops() {
        let t = encode_framed("ab");
        assert_eq!((t[0].id, t[3].id), (BOS, EOS));
        assert_eq!(decode(&t), "ab");
        assert_eq!(t[3].offset, 2);
    }

    #[test]
    fn empty_input() {
        assert!(encode("").is_empty());
        assert_eq!(encode_framed("").len(), 2);
    }

    #[test]
    fn malformed_utf8_is_tokenized_and_located() {
        let bytes = b"ok\xff\xfeend\xe2\x82";
        assert_eq!(decode_bytes(&encode_bytes(bytes)), bytes);
        assert_eq!(invalid_utf8_spans(bytes), vec![(2, 3), (3, 4), (7, 9)]);
        assert!(invalid_utf8_spans("valid \u{1F600}".as_bytes()).is_empty());
    }

    #[test]
    fn no_oov_for_every_unicode_scalar_and_random_bytes() {
        for cp in (0..=0x10FFFFu32).step_by(97).chain([0x7f, 0x80, 0x7ff, 0x800, 0xffff, 0x10000, 0x10ffff]) {
            if let Some(c) = char::from_u32(cp) {
                let s = c.to_string();
                assert_eq!(decode(&encode(&s)), s, "U+{cp:X}");
            }
        }
        let mut x = 12345u64;
        for _ in 0..200 {
            let b: Vec<u8> = (0..64)
                .map(|_| {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (x >> 33) as u8
                })
                .collect();
            assert_eq!(decode_bytes(&encode_bytes(&b)), b);
            assert!(encode_bytes(&b).iter().all(|t| (t.id as usize) < BYTE_COUNT));
        }
    }

    #[test]
    fn ascii_and_punctuation_roundtrip() {
        let s = "Hello, world! It's 3.14 -- \"quoted\"; tab\there\r\n";
        assert_eq!(decode(&encode(s)), s);
    }
}
