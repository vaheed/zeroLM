use std::path::PathBuf;
use zerolm_data::contamination::{ngram_hashes, overlap};
use zerolm_data::corpus::{documents, role_of, role_stream, sha256_hex, verify, Role};
use zerolm_data::manifest::Manifest;

fn dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("zerolm-data-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn docs(n: usize, tag: &str) -> Vec<u8> {
    let mut out = Vec::new();
    for i in 0..n {
        out.extend_from_slice(format!("document {tag} number {i} has its own words {}", i * 7919 % 1013).as_bytes());
        out.push(30);
    }
    out
}

fn manifest(shard: &[u8], extra: &str) -> String {
    format!(
        r#"{{"version":1,"document_separator":30,"splits":{{"exposure":80,"probe":5,"dev":5,"hidden":10}},"split_seed":7,
"sources":[{{"id":"books","description":"d","license":"public domain","license_url":"u","origin_url":"o","generated_by_model":false,"benchmark":false{extra},
"shards":[{{"path":"books.bin","sha256":"{}","bytes":{}}}]}}],"exposure_order":["books"],"evaluation_sets":[]}}"#,
        sha256_hex(shard),
        shard.len()
    )
}

#[test]
fn manifest_rejects_missing_license_generated_text_and_benchmarks() {
    let shard = docs(3, "a");
    assert!(Manifest::parse(&manifest(&shard, "")).is_ok());
    let ok = manifest(&shard, "");
    assert!(Manifest::parse(&ok.replace("\"public domain\"", "\"\"")).is_err());
    assert!(Manifest::parse(&ok.replace("\"generated_by_model\":false", "\"generated_by_model\":true")).is_err());
    assert!(Manifest::parse(&ok.replace("\"benchmark\":false", "\"benchmark\":true")).is_err());
    assert!(Manifest::parse(&ok.replace("\"exposure\":80", "\"exposure\":70")).is_err());
    assert!(Manifest::parse(&ok.replace("books.bin", "../books.bin")).is_err());
    assert!(Manifest::parse(&ok.replace("\"exposure_order\":[\"books\"]", "\"exposure_order\":[\"nope\"]")).is_err());
    assert!(Manifest::parse(&ok.replace(&sha256_hex(&shard), "ABC")).is_err());
}

#[test]
fn verification_catches_tampered_and_missing_shards() {
    let shard = docs(5, "b");
    let m = Manifest::parse(&manifest(&shard, "")).unwrap();
    let root = dir("verify");
    assert!(verify(&m, &root)[0].contains("books.bin"), "missing file");
    std::fs::write(root.join("books.bin"), &shard).unwrap();
    assert!(verify(&m, &root).is_empty());
    let mut bad = shard.clone();
    bad[3] ^= 1;
    std::fs::write(root.join("books.bin"), &bad).unwrap();
    assert!(verify(&m, &root)[0].contains("SHA-256"));
}

#[test]
fn roles_are_deterministic_whole_document_and_near_the_declared_fractions() {
    let shard = docs(4000, "c");
    let m = Manifest::parse(&manifest(&shard, "")).unwrap();
    let mut count = std::collections::HashMap::new();
    for i in 0..4000 {
        assert_eq!(role_of(&m, "books", i), role_of(&m, "books", i));
        *count.entry(role_of(&m, "books", i)).or_insert(0usize) += 1;
    }
    for (role, pct) in [(Role::Exposure, 80.0), (Role::Probe, 5.0), (Role::Dev, 5.0), (Role::Hidden, 10.0)] {
        let got = count[&role] as f64 / 40.0;
        assert!((got - pct).abs() < 2.0, "{role:?}: {got}%");
    }
    // A document keeps its role when later documents are appended.
    let m2 = Manifest::parse(&manifest(&docs(4500, "c"), "")).unwrap();
    assert!((0..4000).all(|i| role_of(&m, "books", i) == role_of(&m2, "books", i)));
}

#[test]
fn the_four_role_streams_partition_the_documents() {
    let shard = docs(600, "d");
    let m = Manifest::parse(&manifest(&shard, "")).unwrap();
    let root = dir("streams");
    std::fs::write(root.join("books.bin"), &shard).unwrap();
    let all = documents(&m, &m.sources[0], &root).unwrap();
    assert_eq!(all.len(), 600);
    let total: usize = [Role::Exposure, Role::Probe, Role::Dev, Role::Hidden].iter().map(|r| role_stream(&m, &root, *r).unwrap().split(|b| *b == b'\n').filter(|l| !l.is_empty()).count()).sum();
    assert_eq!(total, 600, "every document lands in exactly one role");
    assert_eq!(role_stream(&m, &root, Role::Exposure).unwrap(), role_stream(&m, &root, Role::Exposure).unwrap());
}

#[test]
fn contamination_check_finds_a_shared_thirteen_word_window_and_nothing_else() {
    let exposure = b"one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen";
    let index = ngram_hashes(exposure, 13);
    let clean = overlap(&index, b"entirely different words that never appear in the exposure text at all, not even close", 13);
    assert_eq!(clean.shared, 0);
    let leaked = overlap(&index, b"prefix two three four five six seven eight nine ten eleven twelve thirteen fourteen suffix", 13);
    assert_eq!((leaked.windows, leaked.shared), (3, 1));
    let short = overlap(&index, b"two three four", 13);
    assert_eq!(short.windows, 0);
}
