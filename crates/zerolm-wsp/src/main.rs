//! `zerolm-wsp check` compiles every circuit in `model/prog/` and compares the
//! output hash with `model/prog/pinned.json`; `zerolm-wsp hashes` prints them.

fn main() {
    let cmd = std::env::args().nth(1).unwrap_or_else(|| "check".into());
    let rows = match zerolm_wsp::check_pins() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("zerolm-wsp: {e}");
            std::process::exit(2);
        }
    };
    let mut bad = 0;
    for (id, sha, ok) in &rows {
        println!("{id} {sha} {}", if *ok { "pinned" } else { "MISMATCH" });
        bad += !ok as usize;
    }
    if cmd == "check" && bad > 0 {
        eprintln!("zerolm-wsp: {bad} circuit(s) differ from model/prog/pinned.json; re-pin only for a deliberate change");
        std::process::exit(1);
    }
}
