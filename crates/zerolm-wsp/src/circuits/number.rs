//! Fractional-power number circuits: decimal bytes to a number phasor, and a
//! noisy number phasor back to an integer by cleanup.
//!
//! A number is `number(n)_k = exp(i n theta_k)` (seeded base `theta`), so
//! `number(a) * number(b) = number(a + b)`. Because the encoding is a
//! homomorphism, a decimal numeral is the product of its digit phasors:
//! `number(sum_q v_q 10^q) = prod_q number(v_q 10^q)`. The encoder therefore
//! needs no trigonometry: it selects one row per digit from a seeded table
//! (`core.reasoner.number_digits`, row `10 q + v` = `number(v 10^q)`) and binds
//! them.
//!
//! `number-encode`: input `numeral` is 1 x L byte values, the decimal numeral
//! (optionally with a leading `-`) at positions 0..len and zero elsewhere. The
//! host copies bytes and inspects none.
//! 1. Classify each byte against the ten digit codes (one-hot `L x 10`).
//! 2. A digit's place is the number of digit bytes after it (a strict lower
//!    triangular matrix), one-hot over `places`.
//! 3. The outer product of place and digit one-hots selects one table row per
//!    byte; non-digit bytes take the identity phasor (number 0).
//! 4. Bind the L rows. A leading `-` (byte 45) conjugates the result.
//! A digit beyond `places` selects no row, which zeroes the output.
//!
//! `number-decode`: input `v` (1 x 2d). Scores against the seeded codebook
//! (`core.reasoner.number_codebook`, row j = `number(j - range)`), top-1,
//! then the selected row's value (`value`, exact integer) and the codebook row
//! itself (`clean`), so arithmetic can continue on clean vectors.
//!
//! Blocks (units of ablation): encode `classify`, `position`, `select`,
//! `fill`, `product`, `sign`; decode `readout`.

use crate::compiler::{Builder, Compiled};
use serde::{Deserialize, Serialize};
use zerolm_engine::isa::{Axis, GateFn, Instr, IsaError, ReduceFn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncodeParams {
    /// Input width L in bytes.
    pub numeral_bytes: usize,
    /// Decimal places in the digit table (largest magnitude 10^places - 1).
    pub places: usize,
    /// Complex dimension of the output vector.
    pub dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeParams {
    /// The codebook covers [-range, range].
    pub range: u32,
    pub dim: usize,
}

pub const ENCODE_OUTPUTS: [&str; 1] = ["number"];
pub const DECODE_OUTPUTS: [&str; 2] = ["value", "clean"];

/// Byte value of the digit 0 and of the minus sign.
const ZERO: u8 = b'0';
const MINUS: u8 = b'-';

fn onehot(n: usize, i: usize) -> Vec<f32> {
    let mut v = vec![0.0; n];
    v[i] = 1.0;
    v
}

pub fn compile_encode(id: &str, p: &EncodeParams) -> Result<Compiled, IsaError> {
    if p.numeral_bytes < 2 || p.places == 0 || p.places > 9 || p.dim == 0 {
        return Err(IsaError(format!("{id}: invalid number_encode parameters")));
    }
    let (l, q, d) = (p.numeral_bytes, p.places, p.dim);
    let mut b = Builder::new(id);
    b.input("numeral", Some(1), l);
    b.external("digits", 10 * q, 2 * d);

    // Digit one-hot per byte: codes[v] == byte.
    b.block("classify");
    b.constant("codes", 1, 10, (0..10).map(|v| (ZERO + v) as f32).collect());
    b.op(Instr::Transpose { x: "numeral".into(), out: "col".into() });
    b.gate(GateFn::Ge, "col", "codes", "ge_lo");
    b.gate(GateFn::Ge, "codes", "col", "ge_hi");
    b.gate(GateFn::Mul, "ge_lo", "ge_hi", "digit_oh");
    b.reduce(ReduceFn::Sum, "digit_oh", Axis::Rows, "is_digit");

    // Place of each byte: digits after it, one-hot over the table's places.
    b.block("position");
    let mut after = vec![0.0f32; l * l];
    for i in 0..l {
        for j in 0..i {
            after[i * l + j] = 1.0;
        }
    }
    b.constant("after", l, l, after);
    b.op(Instr::Transpose { x: "is_digit".into(), out: "digit_row".into() });
    b.matmul_ab("digit_row", "after", "place_row");
    b.op(Instr::Transpose { x: "place_row".into(), out: "place".into() });
    b.constant("places", 1, q, (0..q).map(|x| x as f32).collect());
    b.gate(GateFn::Ge, "place", "places", "pl_lo");
    b.gate(GateFn::Ge, "places", "place", "pl_hi");
    b.gate(GateFn::Mul, "pl_lo", "pl_hi", "place_oh");

    // Row 10 q + v of the table for a digit v at place q.
    b.block("select");
    let mut spread = vec![0.0f32; q * 10 * q];
    let mut tile = vec![0.0f32; 10 * 10 * q];
    for pq in 0..q {
        for v in 0..10 {
            spread[pq * 10 * q + pq * 10 + v] = 1.0;
            tile[v * 10 * q + pq * 10 + v] = 1.0;
        }
    }
    b.constant("spread", q, 10 * q, spread);
    b.constant("tile", 10, 10 * q, tile);
    b.matmul_ab("place_oh", "spread", "by_place");
    b.matmul_ab("digit_oh", "tile", "by_digit");
    b.gate(GateFn::Mul, "by_place", "by_digit", "sel");
    b.matmul_ab("sel", "digits", "rows");

    // Non-digit bytes are the identity phasor (number 0).
    b.block("fill");
    b.constant("ones_col", l, 1, vec![1.0; l]);
    b.constant("identity", 1, 2 * d, (0..2 * d).map(|k| if k < d { 1.0 } else { 0.0 }).collect());
    b.gate(GateFn::Sub, "ones_col", "is_digit", "non_digit");
    b.matmul_ab("non_digit", "identity", "fill");
    b.gate(GateFn::Add, "rows", "fill", "factors");

    // Product of the L factors.
    b.block("product");
    for j in 0..l {
        b.constant(&format!("pick{j}"), 1, l, onehot(l, j));
        b.matmul_ab(&format!("pick{j}"), "factors", &format!("f{j}"));
    }
    let mut acc = "f0".to_string();
    for j in 1..l {
        let out = format!("p{j}");
        b.op(Instr::Bind { a: acc, b: format!("f{j}"), out: out.clone(), conj: false });
        acc = out;
    }

    // A leading minus conjugates: im *= 1 - 2 neg.
    b.block("sign");
    b.constant("first", l, 1, onehot(l, 0));
    b.scalar("minus", MINUS as f32);
    b.constant("flip", 1, 2 * d, (0..2 * d).map(|k| if k < d { 0.0 } else { -2.0 }).collect());
    b.constant("ones_row", 1, 2 * d, vec![1.0; 2 * d]);
    b.matmul_ab("numeral", "first", "lead");
    b.gate(GateFn::Ge, "lead", "minus", "m_lo");
    b.gate(GateFn::Ge, "minus", "lead", "m_hi");
    b.gate(GateFn::Mul, "m_lo", "m_hi", "neg");
    b.gate(GateFn::Mul, "neg", "flip", "flip_neg");
    b.gate(GateFn::Add, "ones_row", "flip_neg", "sign_mask");
    b.gate(GateFn::Mul, &acc, "sign_mask", "number");

    Ok(b.finish(&ENCODE_OUTPUTS))
}

pub fn compile_decode(id: &str, p: &DecodeParams) -> Result<Compiled, IsaError> {
    if p.range == 0 || p.range > 1 << 23 || p.dim == 0 {
        return Err(IsaError(format!("{id}: invalid number_decode parameters")));
    }
    let r = p.range as i64;
    let m = (2 * r + 1) as usize;
    let mut b = Builder::new(id);
    b.input("v", Some(1), 2 * p.dim);
    b.external("numbers", m, 2 * p.dim);
    b.block("readout");
    b.constant("values", m, 1, (-r..=r).map(|n| n as f32).collect());
    b.matvec("numbers", "v", "scores", false);
    b.op(Instr::Topk { x: "scores".into(), k: 1, idx: None, val: None, mask: Some("best".into()) });
    b.matmul_ab("best", "values", "value");
    b.matmul_ab("best", "numbers", "clean");
    Ok(b.finish(&DECODE_OUTPUTS))
}
