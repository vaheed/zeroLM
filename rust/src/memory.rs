//! Memory with a lifespan: bounded capacity, per-turn decay, reinforcement on
//! recall, and eviction of the weakest entry. No gradients, no training: every
//! change is an explicit write or a fixed-rule update.

use crate::contains_phrase;
use std::fs;
use std::io::Write;
use std::path::Path;

pub const DEFAULT_CAPACITY: usize = 64;
/// Multiplicative strength decay applied once per turn.
pub const DECAY: f64 = 0.97;
/// Entries weaker than this are forgotten.
pub const FORGET_BELOW: f64 = 0.05;
/// Strength added on recall (capped at 1.0).
pub const REINFORCE: f64 = 0.5;
const MAX_FIELD: usize = 512;

#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    /// Normalized key phrase (lowercase, punctuation stripped).
    pub key: String,
    pub value: String,
    pub strength: f64,
    pub last_used: u64,
}

#[derive(Debug, Clone)]
pub struct Memory {
    pub capacity: usize,
    pub clock: u64,
    pub entries: Vec<Entry>,
}

/// Remove tabs/newlines (file format separators), trim, and bound the length.
fn clean(s: &str) -> String {
    let s: String = s
        .chars()
        .map(|c| if matches!(c, '\t' | '\r' | '\n') { ' ' } else { c })
        .collect();
    let s = s.trim();
    let mut end = s.len().min(MAX_FIELD);
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

impl Memory {
    pub fn new(capacity: usize) -> Memory {
        Memory { capacity: capacity.max(1), clock: 0, entries: Vec::new() }
    }

    /// Advance one turn: everything fades a little; the faintest are forgotten.
    pub fn tick(&mut self) {
        self.clock += 1;
        for e in &mut self.entries {
            e.strength *= DECAY;
        }
        self.entries.retain(|e| e.strength >= FORGET_BELOW);
    }

    /// Store or overwrite. Returns the key of an evicted entry, if capacity forced one out.
    pub fn remember(&mut self, key: &str, value: &str) -> Option<String> {
        let key = clean(key);
        let value = clean(value);
        if let Some(e) = self.entries.iter_mut().find(|e| e.key == key) {
            e.value = value;
            e.strength = 1.0;
            e.last_used = self.clock;
            return None;
        }
        self.entries.push(Entry { key, value, strength: 1.0, last_used: self.clock });
        if self.entries.len() > self.capacity {
            return self.evict_weakest();
        }
        None
    }

    fn evict_weakest(&mut self) -> Option<String> {
        let idx = self
            .entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.strength
                    .partial_cmp(&b.strength)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.last_used.cmp(&b.last_used))
            })
            .map(|(i, _)| i)?;
        Some(self.entries.remove(idx).key)
    }

    /// Find the entry whose key appears (as whole words) in `query`. Longest key
    /// wins, then strongest. Recall reinforces the entry. Returns (key, value).
    pub fn recall(&mut self, query: &[u8]) -> Option<(String, String)> {
        let clock = self.clock;
        let best = self
            .entries
            .iter_mut()
            .filter(|e| contains_phrase(query, e.key.as_bytes()))
            .max_by(|a, b| {
                a.key
                    .len()
                    .cmp(&b.key.len())
                    .then(a.strength.partial_cmp(&b.strength).unwrap_or(std::cmp::Ordering::Equal))
            })?;
        best.strength = (best.strength + REINFORCE).min(1.0);
        best.last_used = clock;
        Some((best.key.clone(), best.value.clone()))
    }

    pub fn forget(&mut self, key: &str) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.key != key);
        self.entries.len() != before
    }

    /// Atomic save: write a temp file next to `path`, then rename over it.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut out = format!("LIFE 1\nCLOCK {}\n", self.clock);
        for e in &self.entries {
            out.push_str(&format!("MEM {:.6} {} {}\t{}\n", e.strength, e.last_used, e.key, e.value));
        }
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".tmp");
        let tmp = std::path::PathBuf::from(tmp);
        {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(out.as_bytes())?;
            f.sync_all()?;
        }
        fs::rename(&tmp, path)
    }

    /// Load a life file; a missing file is a newborn (empty memory).
    pub fn load(path: &Path, capacity: usize) -> Result<Memory, String> {
        let mut mem = Memory::new(capacity);
        let text = match fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(mem),
            Err(e) => return Err(format!("cannot read life file {}: {e}", path.display())),
        };
        let mut header = false;
        for (i, line) in text.lines().enumerate() {
            let n = i + 1;
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if !header {
                if line != "LIFE 1" {
                    return Err(format!("invalid life header at line {n}"));
                }
                header = true;
            } else if let Some(rest) = line.strip_prefix("CLOCK ") {
                mem.clock = rest.trim().parse().map_err(|_| format!("invalid CLOCK at line {n}"))?;
            } else if let Some(rest) = line.strip_prefix("MEM ") {
                let bad = || format!("invalid MEM at line {n}");
                let mut it = rest.splitn(3, ' ');
                let strength: f64 = it.next().and_then(|s| s.parse().ok()).ok_or_else(bad)?;
                let last_used: u64 = it.next().and_then(|s| s.parse().ok()).ok_or_else(bad)?;
                let (key, value) = it.next().and_then(|s| s.split_once('\t')).ok_or_else(bad)?;
                if !strength.is_finite() || key.is_empty() {
                    return Err(bad());
                }
                mem.entries.push(Entry {
                    key: key.to_string(),
                    value: value.to_string(),
                    strength: strength.clamp(0.0, 1.0),
                    last_used,
                });
            } else {
                return Err(format!("unknown record at line {n}"));
            }
        }
        if !header {
            return Err("life file is empty or missing header".into());
        }
        while mem.entries.len() > mem.capacity {
            mem.evict_weakest();
        }
        Ok(mem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remember_overwrite_and_recall() {
        let mut m = Memory::new(8);
        m.remember("my name", "Sam");
        m.remember("my name", "Alex");
        assert_eq!(m.entries.len(), 1);
        assert_eq!(m.recall(b"what is my name").unwrap().1, "Alex");
        assert!(m.recall(b"what is the weather").is_none());
    }

    #[test]
    fn evicts_weakest_at_capacity() {
        let mut m = Memory::new(2);
        m.remember("a", "1");
        m.tick();
        m.remember("b", "2");
        m.tick();
        let evicted = m.remember("c", "3");
        assert_eq!(evicted.as_deref(), Some("a")); // oldest = weakest
        assert_eq!(m.entries.len(), 2);
    }

    #[test]
    fn fades_without_use_and_recall_keeps_alive() {
        let mut m = Memory::new(8);
        m.remember("kept", "x");
        m.remember("lost", "y");
        for _ in 0..120 {
            m.tick();
            m.recall(b"kept");
        }
        assert!(m.entries.iter().any(|e| e.key == "kept"));
        assert!(!m.entries.iter().any(|e| e.key == "lost"));
    }

    #[test]
    fn save_load_roundtrip_and_missing_file() {
        let dir = std::env::temp_dir().join(format!("zerolm-mem-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("life.zlm");
        assert!(Memory::load(&p, 8).unwrap().entries.is_empty());
        let mut m = Memory::new(8);
        m.remember("my dog", "is called Rex\tx");
        m.tick();
        m.save(&p).unwrap();
        let back = Memory::load(&p, 8).unwrap();
        assert_eq!(back.clock, 1);
        assert_eq!(back.entries.len(), 1);
        assert_eq!(back.entries[0].value, "is called Rex x");
        fs::write(&p, "garbage\n").unwrap();
        assert!(Memory::load(&p, 8).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }
}
