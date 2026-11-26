//! Seed-level statistics for the GDM experiment: Student-t intervals over seed-paired differences and
//! Cohen's d_z. No external dependency; the t quantile inverts the regularised incomplete beta function.

fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation.
    const C: [f64; 6] = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let mut y = x;
    let tmp = x + 5.5 - (x + 0.5) * (x + 5.5).ln();
    let mut ser = 1.000000000190015;
    for c in C {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-300 {
        d = 1e-300;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..300 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-300 { d = 1e-300; }
        c = 1.0 + aa / c;
        if c.abs() < 1e-300 { c = 1e-300; }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-300 { d = 1e-300; }
        c = 1.0 + aa / c;
        if c.abs() < 1e-300 { c = 1e-300; }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-14 {
            break;
        }
    }
    h
}

/// Regularised incomplete beta function I_x(a, b).
pub fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) { bt * betacf(a, b, x) / a } else { 1.0 - bt * betacf(b, a, 1.0 - x) / b }
}

/// CDF of Student's t with `df` degrees of freedom.
pub fn t_cdf(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    let tail = 0.5 * betai(df / 2.0, 0.5, x);
    if t >= 0.0 { 1.0 - tail } else { tail }
}

/// Quantile of Student's t (bisection).
pub fn t_quantile(p: f64, df: f64) -> f64 {
    let (mut lo, mut hi) = (-1e3, 1e3);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if t_cdf(mid, df) < p { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}

#[derive(Debug, Clone)]
pub struct Interval {
    pub n: usize,
    pub mean: f64,
    pub sd: f64,
    pub lo: f64,
    pub hi: f64,
    /// Cohen's d_z = mean / sd of the paired differences.
    pub dz: f64,
    pub level: f64,
}

/// Two-sided t interval of the mean of `xs` at confidence `level` (for example 0.95).
pub fn t_interval(xs: &[f64], level: f64) -> Interval {
    let n = xs.len();
    let mean = xs.iter().sum::<f64>() / n.max(1) as f64;
    let sd = if n > 1 { (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt() } else { f64::NAN };
    let half = if n > 1 { t_quantile(1.0 - (1.0 - level) / 2.0, n as f64 - 1.0) * sd / (n as f64).sqrt() } else { f64::NAN };
    Interval { n, mean, sd, lo: mean - half, hi: mean + half, dz: if sd > 0.0 { mean / sd } else { f64::NAN }, level }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_quantiles_match_tables() {
        for (p, df, want) in [(0.975, 4.0, 2.7764), (0.975, 7.0, 2.3646), (0.975, 2.0, 4.3027), (0.995, 9.0, 3.2498), (0.95, 30.0, 1.6973)] {
            assert!((t_quantile(p, df) - want).abs() < 2e-3, "t({p},{df}) = {} want {want}", t_quantile(p, df));
        }
    }

    #[test]
    fn interval_of_known_sample() {
        let i = t_interval(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.95);
        assert!((i.mean - 3.0).abs() < 1e-12 && (i.sd - 2.5f64.sqrt()).abs() < 1e-12);
        assert!((i.hi - 3.0 - 2.7764 * 2.5f64.sqrt() / 5f64.sqrt()).abs() < 5e-3);
    }
}
