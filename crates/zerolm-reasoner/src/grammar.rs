//! Host row routing for the generic lookup and sequence-rewrite circuits.

use crate::Reasoner;
use crate::circuits::with_consts;
use crate::grammar_seed::{KEY_BYTES, WINDOW};
use zerolm_engine::isa::{Buf, View};

impl Reasoner {
    /// Model computation runs in the circuits; the host moves the rows they
    /// select. Every rewrite uses an input-length step budget, independent
    /// of matches. Inputs are not bounded by the four-token window.
    pub(crate) fn normalized_tokens(&self, text: &str) -> Vec<String> {
        let original = self.language_tokens(text);
        let n = original.len();
        // Empty input requires no model decisions or tensor operations.
        if n == 0 {
            return original;
        }
        let g = &self.grammar;
        let keys: Vec<f32> = original.iter().flat_map(|word| {
            let mut row = vec![0.0; KEY_BYTES];
            for (dst, src) in row.iter_mut().zip(word.bytes()) {
                *dst = src as f32;
            }
            row
        }).collect();
        let key_table = &g.tensors[0];
        let values = &g.tensors[1];
        let lookup_weights = with_consts(&g.lookup, vec![
            ("table_keys", View::new(key_table.rows, key_table.cols, &key_table.data)),
            ("table_values", View::new(values.rows, values.cols, &values.data)),
        ]);
        let classified = self.exec(&g.lookup, vec![("keys", Buf::new(n, KEY_BYTES, keys))], &lookup_weights);
        let mut ids = classified.get("values").data.clone();
        let mut origins: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let pool: Vec<&str> = original.iter().chain(&g.words).map(String::as_str).collect();
        let mut length = n;
        for rule in 0..g.tensors[2].rows {
            let weights = with_consts(&g.rewrite, vec![
                ("from_ids", g.row(2, rule)), ("from_mask", g.row(3, rule)),
                ("to_ids", g.row(4, rule)), ("to_mask", g.row(5, rule)),
                ("at_end", g.row(6, rule)),
            ]);
            ids.resize(length + WINDOW, 0.0);
            origins.resize(length + WINDOW, 0.0);
            let mut next_ids = Vec::new();
            let mut next_origins = Vec::new();
            let mut cursor = Buf::row(vec![0.0]);
            for _ in 0..length {
                let start = cursor.data[0] as usize;
                let out = self.exec(&g.rewrite, vec![
                    ("ids", Buf::row(ids[start..start + WINDOW].to_vec())),
                    ("origins", Buf::row(origins[start..start + WINDOW].to_vec())),
                    ("cursor", cursor), ("length", Buf::row(vec![length as f32])),
                    ("base", Buf::row(vec![n as f32])),
                ], &weights);
                let count = out.get("count").data[0] as usize;
                next_ids.extend_from_slice(&out.get("output_ids").data[..count]);
                next_origins.extend_from_slice(&out.get("output_origins").data[..count]);
                cursor = out.get("next_cursor").clone();
            }
            ids = next_ids;
            origins = next_origins;
            length = ids.len();
        }
        origins.iter().map(|index| pool[*index as usize].to_string()).collect()
    }
}

#[cfg(test)]
#[path = "grammar_tests.rs"]
mod tests;
