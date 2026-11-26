//! 13-word-window overlap of each held-out role with the first 10^8 exposure bytes (real-text curves).
use zerolm_data::{contamination, corpus::*, manifest::Manifest};
fn main() {
    let m = Manifest::parse(&std::fs::read_to_string("research/corpus/manifest.json").unwrap()).unwrap();
    let r = std::path::Path::new("data");
    let mut e = Vec::new();
    for id in &m.exposure_order {
        e.extend(source_role_stream(&m, r, id, Role::Exposure).unwrap());
    }
    e.truncate(100_000_000);
    let idx = contamination::ngram_hashes(&e, 13);
    for (name, id, n) in [("fiction", "gutenberg-books", usize::MAX), ("nonfiction", "gutenberg-nonfiction", 1_000_000), ("wikipedia", "simple-wikipedia", 2_000_000)] {
        let mut h = source_role_stream(&m, r, id, Role::Hidden).unwrap();
        h.truncate(n);
        let o = contamination::overlap(&idx, &h, 13);
        println!("{name}: {}/{} 13-word windows ({:.2}%) occur in the first 10^8 exposure bytes", o.shared, o.windows, 100.0 * o.shared as f64 / o.windows as f64);
    }
}
