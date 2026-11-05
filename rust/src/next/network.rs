//! Dense tensors and the plain-text weight file (`weights.zln`). Generic arithmetic only: no word,
//! role or grammar knowledge lives here.

use std::collections::HashMap;
use std::fs;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Tensor {
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// y = W x
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.cols);
        (0..self.rows).map(|i| dot(self.row(i), x)).collect()
    }

    pub fn nonzero(&self) -> usize {
        self.data.iter().filter(|v| **v != 0.0).count()
    }
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// h = relu(W z + b)
pub fn dense_relu(w: &Tensor, b: &Tensor, z: &[f64]) -> Vec<f64> {
    let mut h = w.matvec(z);
    for (v, bias) in h.iter_mut().zip(&b.data) {
        *v = relu(*v + bias);
    }
    h
}

#[derive(Debug, Clone)]
pub struct Net {
    pub tensors: HashMap<String, Tensor>,
    pub lists: HashMap<String, Vec<String>>,
    pub meta: HashMap<String, String>,
}

impl Net {
    pub fn load(path: &str) -> Result<Net, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        Net::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Net, String> {
        let mut lines = text.lines();
        if lines.next() != Some("ZLN 1") {
            return Err("not a ZLN 1 weight file".into());
        }
        let mut net = Net { tensors: HashMap::new(), lists: HashMap::new(), meta: HashMap::new() };
        while let Some(line) = lines.next() {
            let mut w = line.split_whitespace();
            match w.next() {
                Some("meta") => {
                    let k = w.next().ok_or("meta without key")?;
                    net.meta.insert(k.to_string(), w.collect::<Vec<_>>().join(" "));
                }
                Some("list") => {
                    let name = w.next().ok_or("list without name")?.to_string();
                    let n: usize = w.next().and_then(|v| v.parse().ok()).ok_or("list without count")?;
                    let items: Vec<String> = w.map(str::to_string).collect();
                    if items.len() != n {
                        return Err(format!("list {name}: expected {n} items, found {}", items.len()));
                    }
                    net.lists.insert(name, items);
                }
                Some("tensor") => {
                    let name = w.next().ok_or("tensor without name")?.to_string();
                    let rows: usize = w.next().and_then(|v| v.parse().ok()).ok_or("tensor rows")?;
                    let cols: usize = w.next().and_then(|v| v.parse().ok()).ok_or("tensor cols")?;
                    let mut data = Vec::with_capacity(rows * cols);
                    for r in 0..rows {
                        let row = lines.next().ok_or(format!("tensor {name}: missing row {r}"))?;
                        let vals: Result<Vec<f64>, _> = row.split_whitespace().map(str::parse::<f64>).collect();
                        let vals = vals.map_err(|e| format!("tensor {name} row {r}: {e}"))?;
                        if vals.len() != cols {
                            return Err(format!("tensor {name} row {r}: {} values, expected {cols}", vals.len()));
                        }
                        if vals.iter().any(|v| !v.is_finite()) {
                            return Err(format!("tensor {name} row {r}: non-finite value"));
                        }
                        data.extend(vals);
                    }
                    net.tensors.insert(name, Tensor { rows, cols, data });
                }
                None => {}
                Some(other) => return Err(format!("unknown record {other}")),
            }
        }
        Ok(net)
    }

    pub fn t(&self, name: &str) -> &Tensor {
        // Presence and shape of every tensor is checked once in `NextModel::from_net`.
        &self.tensors[name]
    }

    pub fn list(&self, name: &str) -> &[String] {
        &self.lists[name]
    }

    pub fn dim(&self, key: &str) -> usize {
        self.meta.get(key).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0) as usize
    }

    pub fn param_count(&self) -> (usize, usize) {
        let total: usize = self.tensors.values().map(|t| t.rows * t.cols).sum();
        let nz: usize = self.tensors.values().map(Tensor::nonzero).sum();
        (total, nz)
    }
}
