use crate::manifest::Forbidden;
use crate::Violation;
use std::path::Path;

const SKIP_DIRS: [&str; 4] = [".git", "target", "node_modules", ".claude"];
const MAX_SCAN_BYTES: u64 = 2 * 1024 * 1024;

/// Walk the tree; flag pretrained-weight extensions anywhere, and
/// model-download substrings in non-exempt text files.
pub fn scan_tree(root: &Path, f: &Forbidden) -> Vec<Violation> {
    let mut out = Vec::new();
    walk(root, root, f, &mut out);
    out
}

fn walk(root: &Path, dir: &Path, f: &Forbidden, out: &mut Vec<Violation>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let path = e.path();
        let rel = path.strip_prefix(root).unwrap_or(&path).to_string_lossy().replace('\\', "/");
        let Ok(ft) = e.file_type() else { continue };
        if ft.is_dir() {
            if !SKIP_DIRS.iter().any(|s| e.file_name() == *s) {
                walk(root, &path, f, out);
            }
            continue;
        }
        if !ft.is_file() {
            continue;
        }
        let lower = rel.to_lowercase();
        if let Some(ext) = f.extensions.iter().find(|x| lower.ends_with(x.as_str())) {
            out.push(Violation::new("forbidden-artifacts", format!("{rel}: pretrained-weight extension '{ext}'")));
            continue;
        }
        if f.exempt_prefixes.iter().any(|p| rel.starts_with(p.as_str())) {
            continue;
        }
        let Ok(meta) = e.metadata() else { continue };
        if meta.len() > MAX_SCAN_BYTES {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        for s in &f.substrings {
            if text.contains(s.as_str()) {
                out.push(Violation::new("forbidden-artifacts", format!("{rel}: contains '{s}'")));
            }
        }
    }
}
