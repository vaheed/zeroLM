use zerolm_data::{corpus::*, manifest::Manifest};
use zerolm_eval::representation::{kgram_index, novel_positions};
fn main() {
    let m = Manifest::parse(&std::fs::read_to_string("research/corpus/manifest.json").unwrap()).unwrap();
    let r = std::path::Path::new("data");
    let cap = |mut v: Vec<u8>, n: usize| { v.truncate(n); v };
    let e = [cap(source_role_stream(&m, r, "gutenberg-books", Role::Exposure).unwrap(), 5_000_000), cap(source_role_stream(&m, r, "gutenberg-nonfiction", Role::Exposure).unwrap(), 5_000_000)].concat();
    let dev = [cap(source_role_stream(&m, r, "gutenberg-books", Role::Dev).unwrap(), 1_000_000), cap(source_role_stream(&m, r, "gutenberg-nonfiction", Role::Dev).unwrap(), 1_000_000)].concat();
    for k in [6usize, 8, 10, 12, 16, 24] {
        let idx = kgram_index(&e, k);
        let f = novel_positions(&dev, &idx, k).iter().filter(|b| **b).count() as f64 / dev.len() as f64;
        println!("k {k}: {:.1}% of dev positions have a context absent from 10^7 exposure bytes", f * 100.0);
    }
}
