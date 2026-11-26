//! Frozen language-model level thresholds (`model/specs/levels.json`, roadmap section 7.1).

use serde_json::Value;
use std::path::Path;

pub const SPEC: &str = "model/specs/levels.json";

pub struct Levels(pub Value);

impl Levels {
    pub fn load(root: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(root.join(SPEC)).map_err(|e| format!("{SPEC}: {e}"))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{SPEC}: {e}"))?;
        let l = Levels(v);
        l.validate()?;
        Ok(l)
    }

    fn num(&self, level: &str, key: &str) -> Option<f64> {
        self.0["levels"][level][key].as_f64()
    }

    /// Thresholds must exist and rise from one level to the next.
    pub fn validate(&self) -> Result<(), String> {
        for l in ["L1", "L2", "L3", "L4"] {
            if self.0["levels"][l]["name"].as_str().is_none() {
                return Err(format!("level {l} missing"));
            }
        }
        for (l, k) in [("L2", "blimp_min"), ("L3", "blimp_min"), ("L3", "lambada_accuracy_min"), ("L2", "wikipedia_heldout_bpb_max"), ("L3", "wikipedia_heldout_bpb_max")] {
            let v = self.num(l, k).ok_or(format!("{l}.{k} missing"))?;
            if !(v > 0.0 && v.is_finite()) {
                return Err(format!("{l}.{k} out of range"));
            }
        }
        if self.num("L3", "blimp_min").unwrap() <= self.num("L2", "blimp_min").unwrap() {
            return Err("L3 BLiMP must exceed L2".into());
        }
        if self.num("L3", "wikipedia_heldout_bpb_max").unwrap() >= self.num("L2", "wikipedia_heldout_bpb_max").unwrap() {
            return Err("L3 BPB bound must be below L2".into());
        }
        if self.num("L3", "blimp_min").unwrap() > 1.0 || self.num("L3", "lambada_accuracy_min").unwrap() > 1.0 {
            return Err("accuracies are fractions in (0, 1]".into());
        }
        Ok(())
    }

    /// L1 bar for a model scored at `exposure_bytes`: the better of the two reference
    /// compressors measured on the same stream (`rows`: baseline name, exposure bytes, BPB).
    pub fn l1_bar(rows: &[(String, usize, f64)], exposure_bytes: usize) -> Option<f64> {
        rows.iter().filter(|(n, p, _)| *p == exposure_bytes && (n == "ppm-c-5" || n == "kneser-ney-5")).map(|r| r.2).reduce(f64::min)
    }

    pub fn l1_met(rows: &[(String, usize, f64)], exposure_bytes: usize, bpb: f64) -> Option<bool> {
        Self::l1_bar(rows, exposure_bytes).map(|bar| bpb <= bar)
    }
}

impl Levels {
    /// SHA-256 of the frozen spec file text (compiled in, so the report names the thresholds it was run against).
    pub fn spec_sha256() -> String {
        zerolm_data::corpus::sha256_hex(include_str!("../../../model/specs/levels.json").as_bytes())
    }
}

/// The `reports/real-text/summary.json` document (`research/schemas/real-text-report.schema.json`).
#[allow(clippy::too_many_arguments)]
pub fn realtext_summary(
    rows: &[(String, usize, f64)],
    exposure_sha256: &str,
    hidden_sha256: &str,
    hidden_bytes: usize,
    manifest_sha256: &str,
    levels_sha256: &str,
    l1_bars: Vec<Value>,
    control_included: bool,
) -> Value {
    let rows: Vec<Value> = rows.iter().map(|(n, p, b)| serde_json::json!({"baseline": n, "exposure_bytes": p, "hidden_bpb": b})).collect();
    serde_json::json!({
        "report": "real-text",
        "exposure_sha256": exposure_sha256,
        "hidden_sha256": hidden_sha256,
        "hidden_bytes": hidden_bytes,
        "manifest_sha256": manifest_sha256,
        "levels_sha256": levels_sha256,
        "control_included": control_included,
        "rows": rows,
        "l1_bars": l1_bars,
        "scope_note": "Count-based and CONTROL baselines on one seed and one split; the CONTROL is excluded from strict results. No ZeroLM model is claimed to reach any level by this report."
    })
}
