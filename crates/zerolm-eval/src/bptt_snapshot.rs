//! Minimal Genome and Snapshot for the online learner (`docs/gdm-experiment.md`, reproducibility contract).
//!
//! * **Genome**: a canonical JSON description of the architecture and every tensor: name, offset, length,
//!   class (`parameter`, `optimizer`, `transient_state`, `persistent_fast_weight`), whether an update can change it,
//!   and the initialization provenance (seed, source tags, SHA-256 of the initial parameter vector). Its SHA-256
//!   (`genome_sha256`) changes if any of that changes. No dynamic topology exists.
//! * **Snapshot**: a byte string holding the parameters, the Adam moments and step count, the recurrent state,
//!   the fast-weight state, the byte history, and the prequential counters, closed by a SHA-256 of everything
//!   before it. Restoring it into a model built from the same genome reproduces every later bit of execution.
use super::*;
use serde_json::{json, Value};

const MAGIC: &[u8; 8] = b"ZLMSNAP1";

/// What a model carries between chunks besides its weights.
#[derive(Debug, Clone, PartialEq)]
pub struct CarryState {
    pub h: Vec<f32>,
    pub fast: Vec<f32>,
    pub tail: Vec<u8>,
}

fn sha_hex(bytes: &[u8]) -> String {
    let mut h = zerolm_tensor::sha256::Sha256::new();
    h.update(bytes);
    h.finish_hex()
}

impl Bptt {
    /// A fresh carry state (zero recurrent state, zero fast weights, start byte history).
    pub fn fresh_state(&self) -> CarryState {
        CarryState { h: vec![0.0; self.l.d], fast: vec![0.0; self.fast.len()], tail: vec![START_BYTE] }
    }

    /// The state as the model currently carries it.
    pub fn carry_state(&self) -> CarryState {
        CarryState { h: self.state.clone(), fast: self.fast.clone(), tail: self.tail.clone() }
    }

    /// Bits of each byte of `bytes` continuing from `st`; changes neither the weights nor `st`.
    pub fn score_from(&self, st: &CarryState, bytes: &[u8]) -> Vec<f32> {
        self.probs_from(st, bytes).into_iter().map(|(bits, _)| bits).collect()
    }

    /// Per byte: (bits, argmax byte) continuing from `st`.
    pub fn probs_from(&self, st: &CarryState, bytes: &[u8]) -> Vec<(f32, u8)> {
        let (mut h, mut fs) = (st.h.clone(), st.fast.clone());
        let mut buf = st.tail.clone();
        let t0 = buf.len();
        buf.extend_from_slice(bytes);
        let mut out = Vec::with_capacity(bytes.len());
        for (i, &y) in bytes.iter().enumerate() {
            let s = self.forward_step(&self.p, &h, &fs, &buf[..t0 + i]);
            let am = s.p.iter().enumerate().fold((0usize, 0.0f32), |a, (i, v)| if *v > a.1 { (i, *v) } else { a }).0;
            out.push((-(s.p[y as usize].max(1e-30)).log2(), am as u8));
            h = s.h;
            if let Some(gd) = s.gd {
                fs = gd.s_new;
            }
        }
        out
    }

    /// The state after reading `bytes` from `st` with frozen weights (no update of anything).
    pub fn advance(&self, st: &CarryState, bytes: &[u8]) -> CarryState {
        let (mut h, mut fs) = (st.h.clone(), st.fast.clone());
        let mut buf = st.tail.clone();
        let t0 = buf.len();
        buf.extend_from_slice(bytes);
        for i in 0..bytes.len() {
            let s = self.forward_step(&self.p, &h, &fs, &buf[..t0 + i]);
            h = s.h;
            if let Some(gd) = s.gd {
                fs = gd.s_new;
            }
        }
        // Convention shared with `train_chunk`: the last byte of `tail` is the pending input; the state does not
        // include it yet, and the next call consumes it first.
        let keep = buf.len().saturating_sub(16);
        CarryState { h, fast: fs, tail: buf[keep..].to_vec() }
    }

    /// Replace the carried recurrent/fast-weight state (a "reset" is `fresh_state()` or a field of it).
    pub fn set_carry_state(&mut self, st: &CarryState) {
        self.state = st.h.clone();
        self.fast = st.fast.clone();
        self.tail = st.tail.clone();
        self.prev = *self.tail.last().unwrap_or(&START_BYTE);
    }

    /// Canonical genome description.
    pub fn genome(&self) -> Value {
        let l = &self.l;
        let mut tensors = vec![
            ("embed", l.embed, VOCAB * l.e),
            ("gru.wz", l.w[0], l.d * l.e),
            ("gru.wr", l.w[1], l.d * l.e),
            ("gru.wh", l.w[2], l.d * l.e),
            ("gru.uz", l.u[0], l.d * l.d),
            ("gru.ur", l.u[1], l.d * l.d),
            ("gru.uh", l.u[2], l.d * l.d),
            ("gru.bz", l.b[0], l.d),
            ("gru.br", l.b[1], l.d),
            ("gru.bh", l.b[2], l.d),
            ("readout.weight", l.ro, VOCAB * (l.d + l.e)),
            ("readout.bias", l.rb, VOCAB),
        ];
        let mut state = vec![("gru.h", "transient_state", l.d)];
        let mut init = vec![json!({"tensors": "embed, gru.*", "source": "seeded_prng", "tag_family": "zerolm-engine bootstrap (Model::new)", "seed": self.seed}), json!({"tensors": "readout.*", "source": "zeros"})];
        if let Some(g) = self.cfg.gdm {
            let (r, e) = (g.rank, l.e);
            tensors.extend([("gdm.u", l.gu, VOCAB * r), ("gdm.v", l.gv, VOCAB * r), ("gdm.d", l.gd, VOCAB * r), ("gdm.a0", l.ga0, 1), ("gdm.b0", l.gb0, 1), ("gdm.gamma0", l.gg0, 1), ("gdm.wa", l.gwa, e), ("gdm.wb", l.gwb, e)]);
            state.push(("gdm.S", "persistent_fast_weight", r * r));
            init.push(match g.init {
                GdmInit::Compiled => json!({"tensors": "gdm.u, gdm.v, gdm.d", "source": "compiled_wsp", "construction": "one seeded unit-row code table, tag gdm-code-table-v1, seed 0 (content-independent); U=V=D"}),
                GdmInit::Random => json!({"tensors": "gdm.u, gdm.v, gdm.d", "source": "seeded_prng", "tags": ["gdm-random-u", "gdm-random-v", "gdm-random-d"], "seed": self.seed}),
            });
            init.push(json!({"tensors": "gdm.a0, gdm.b0, gdm.gamma0", "source": "constants", "values": [g.a0, g.b0, g.gamma0]}));
            init.push(json!({"tensors": "gdm.wa, gdm.wb", "source": "zeros"}));
            init.push(json!({"tensors": "position tags (constants, not parameters)", "source": "seeded_prng", "tag": "gdm-position-tags-v1"}));
        }
        let off = self.cfg.gdm.is_some_and(|g| g.off);
        let table: Vec<Value> = tensors.iter().map(|(n, o, len)| json!({"name": n, "offset": o, "len": len, "class": "parameter", "updatable": !(off && n.starts_with("gdm."))})).collect();
        let mut st: Vec<Value> = state.iter().map(|(n, c, len)| json!({"name": n, "class": c, "len": len})).collect();
        st.push(json!({"name": "adam.m", "class": "optimizer", "len": l.total}));
        st.push(json!({"name": "adam.v", "class": "optimizer", "len": l.total}));
        let c = &self.cfg;
        let arch = json!({
            "core": "gru", "embed_dim": l.e, "hidden_dim": l.d, "vocab": VOCAB, "params_total": l.total,
            "window": c.window, "lr": c.lr, "clip": c.clip, "beta1": c.beta1, "beta2": c.beta2,
            "rule": RULE, "signal": SIGNAL,
            "circuit": c.circuit.map(|x| format!("{x:?}")),
            "gdm": c.gdm.map(|g| json!({"rank": g.rank, "lags": g.lags, "init": format!("{:?}", g.init), "a0": g.a0, "b0": g.b0, "gamma0": g.gamma0, "lr_mult": g.lr_mult, "off": g.off})),
        });
        let mut g = json!({"genome_version": 1, "architecture": arch, "tensors": table, "state": st, "initialization": init, "init_params_sha256": self.init_sha256});
        let hash = sha_hex(g.to_string().as_bytes());
        g["genome_sha256"] = json!(hash);
        g
    }

    pub fn genome_sha256(&self) -> String {
        self.genome()["genome_sha256"].as_str().unwrap().to_string()
    }

    /// Serialize everything needed to continue execution exactly.
    pub fn snapshot(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        let head = json!({"genome_sha256": self.genome_sha256(), "adam_steps": self.steps, "seen": self.seen, "bits_bits": self.bits.to_bits(), "prev": self.prev,
            "lens": {"p": self.p.len(), "m": self.m.len(), "v": self.v.len(), "h": self.state.len(), "fast": self.fast.len(), "tail": self.tail.len()}});
        let hb = head.to_string().into_bytes();
        out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
        out.extend_from_slice(&hb);
        for v in [&self.p, &self.m, &self.v, &self.state, &self.fast] {
            for x in v.iter() {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out.extend_from_slice(&self.tail);
        let digest = sha_hex(&out);
        out.extend_from_slice(digest.as_bytes());
        out
    }

    /// SHA-256 of the snapshot bytes (the checkpoint identity).
    pub fn snapshot_sha256(&self) -> String {
        sha_hex(&self.snapshot())
    }

    /// Restore a snapshot into this model, which must have been built from the same genome.
    pub fn restore(&mut self, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() < MAGIC.len() + 8 + 64 || &bytes[..8] != MAGIC {
            return Err("not a ZeroLM snapshot".into());
        }
        let (body, digest) = bytes.split_at(bytes.len() - 64);
        if sha_hex(body).as_bytes() != digest {
            return Err("snapshot hash mismatch (corrupt or truncated)".into());
        }
        let hl = u64::from_le_bytes(body[8..16].try_into().unwrap()) as usize;
        let head: Value = serde_json::from_slice(&body[16..16 + hl]).map_err(|e| e.to_string())?;
        if head["genome_sha256"].as_str() != Some(&self.genome_sha256()) {
            return Err("snapshot was written by a different genome".into());
        }
        let mut at = 16 + hl;
        let mut take_f32 = |n: usize| -> Vec<f32> {
            let v = body[at..at + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            at += 4 * n;
            v
        };
        let n = |k: &str| head["lens"][k].as_u64().unwrap() as usize;
        let (p, m, v, h, fast) = (take_f32(n("p")), take_f32(n("m")), take_f32(n("v")), take_f32(n("h")), take_f32(n("fast")));
        if p.len() != self.p.len() || fast.len() != self.fast.len() || h.len() != self.state.len() {
            return Err("snapshot tensor sizes do not match the genome".into());
        }
        self.tail = body[at..at + n("tail")].to_vec();
        self.p = p;
        self.m = m;
        self.v = v;
        self.state = h;
        self.fast = fast;
        self.steps = head["adam_steps"].as_u64().unwrap();
        self.seen = head["seen"].as_u64().unwrap() as usize;
        self.bits = f64::from_bits(head["bits_bits"].as_u64().unwrap());
        self.prev = head["prev"].as_u64().unwrap() as u8;
        Ok(())
    }
}
