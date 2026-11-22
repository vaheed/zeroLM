//! Host routing for `language-lex`. All byte classification, case folding,
//! span selection, and chunk continuation decisions are circuit outputs.

use crate::Reasoner;
use crate::circuits::with_consts;
use zerolm_engine::isa::Buf;

impl Reasoner {
    pub(crate) fn language_tokens(&self, text: &str) -> Vec<String> {
        let c = &self.circuits.lexer;
        let l = c.program.inputs[0].cols.expect("fixed lexer width");
        let weights = with_consts(c, vec![]);
        let bytes: Vec<u8> = text.bytes().collect();
        let mut tokens: Vec<String> = Vec::new();
        let mut tail = 0.0;
        for chunk in bytes.chunks(l) {
            let mut row = vec![0.0; l];
            for (dst, src) in row.iter_mut().zip(chunk) {
                *dst = *src as f32;
            }
            let out = self.exec(c, vec![("text", Buf::row(row)), ("tail", Buf::row(vec![tail]))], &weights);
            let folded: Vec<u8> = out.get("folded").data.iter().map(|x| *x as u8).collect();
            let count = out.get("count").data[0] as usize;
            for i in 0..count {
                let start = out.get("start").data[i] as usize;
                let end = out.get("end").data[i] as usize + 1;
                let word = std::str::from_utf8(&folded[start..end]).expect("lexer spans contain only ASCII");
                if i == 0 && out.get("join").data[0] != 0.0 {
                    tokens.last_mut().expect("continuation has a preceding token").push_str(word);
                } else {
                    tokens.push(word.to_string());
                }
            }
            tail = out.get("tail_word").data[0];
        }
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ReasonerConfig;
    use zerolm_engine::semantic_state::tokens;

    #[test]
    fn compiled_tokens_match_reference_across_chunks_and_unicode() {
        let r = Reasoner::new(ReasonerConfig { dim: 64, symbols: 32, slots: 32, number_range: 20, ..Default::default() }).unwrap();
        let ascii: String = (0..=127).map(char::from).collect();
        for text in [
            String::new(), ascii, "A-B 12\0Z_é中é Q?\tR\n".into(),
            "a".repeat(513), "!".repeat(513), "A- Z!".repeat(110),
            format!("{}B C", " ".repeat(127)),
            format!("{}!C", "A".repeat(128)),
            format!("{}B", "A ".repeat(64)),
            format!("{}éB", "A".repeat(127)),
        ] {
            assert_eq!(r.language_tokens(&text), tokens(&text), "{text:?}");
        }
        // Seeded arbitrary valid Unicode also exercises UTF-8 split across
        // chunks: every non-ASCII byte must act as a separator.
        let mut seed = 177u64;
        let text: String = (0..4096).map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            char::from_u32(((seed >> 32) % 0x110000) as u32).unwrap_or(' ')
        }).collect();
        assert_eq!(r.language_tokens(&text), tokens(&text));
    }
}
