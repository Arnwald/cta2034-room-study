//! Special functions and numerical helpers: Bessel functions, Gauss-Legendre nodes, a
//! not-a-knot cubic spline, linear interpolation and percentiles.

/// Bessel function of the first kind, order 0.
pub fn j0(x: f64) -> f64 {
    libm::j0(x)
}

/// Bessel function of the first kind, order 1.
pub fn j1(x: f64) -> f64 {
    libm::j1(x)
}

/// ``2 J1(x) / x`` with its limit 1 at ``x = 0``: the far field of a flat piston.
pub fn piston(x: f64) -> f64 {
    if x.abs() > 1e-8 {
        2.0 * j1(x) / x
    } else {
        1.0
    }
}

/// Modified Bessel function of the first kind, order 0 (series), for the Kaiser window.
pub fn i0(x: f64) -> f64 {
    let y = 0.25 * x * x;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..500 {
        term *= y / (k as f64 * k as f64);
        sum += term;
        if term < 1e-17 * sum {
            break;
        }
    }
    sum
}

/// Kaiser window of ``m`` points, as NumPy's ``np.kaiser``.
pub fn kaiser(m: usize, beta: f64) -> Vec<f64> {
    let denom = i0(beta);
    (0..m)
        .map(|n| {
            let r = 2.0 * n as f64 / (m as f64 - 1.0) - 1.0;
            i0(beta * (1.0 - r * r).max(0.0).sqrt()) / denom
        })
        .collect()
}

/// Gauss-Legendre nodes (ascending) and weights on [-1, 1].
pub fn leggauss(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 0.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, z);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            let p = if n == 0 { 1.0 } else if n == 1 { z } else { p1 };
            let pm1 = if n == 1 { 1.0 } else { p0 };
            dp = n as f64 * (z * p - pm1) / (z * z - 1.0);
            let dz = p / dp;
            z -= dz;
            if dz.abs() < 1e-15 {
                break;
            }
        }
        x[n - 1 - i] = z;
        w[n - 1 - i] = 2.0 / ((1.0 - z * z) * dp * dp);
    }
    (x, w)
}

/// Linear interpolation with the ends held, as ``np.interp`` (``xp`` increasing).
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    let i = xp.partition_point(|&v| v <= x);
    let (x0, x1) = (xp[i - 1], xp[i]);
    let t = (x - x0) / (x1 - x0);
    fp[i - 1] + t * (fp[i] - fp[i - 1])
}

/// Percentile with linear interpolation between order statistics (NumPy's default).
pub fn percentile(values: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = q / 100.0 * (v.len() as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let t = pos - lo as f64;
    v[lo] + t * (v[hi] - v[lo])
}

pub fn median(values: &[f64]) -> f64 {
    percentile(values, 50.0)
}

pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Least-squares straight line ``y = a x + b``: returns ``(a, b)``.
pub fn linefit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
    }
    let a = sxy / sxx;
    (a, my - a * mx)
}

/// Cubic spline through ``(x, y)`` with SciPy's default "not-a-knot" end conditions, built as
/// SciPy builds it: the slopes at the knots from a tridiagonal system, then cubic Hermite pieces.
pub struct CubicSpline {
    x: Vec<f64>,
    y: Vec<f64>,
    s: Vec<f64>,
}

impl CubicSpline {
    pub fn new(x: &[f64], y: &[f64]) -> Self {
        let n = x.len();
        assert!(n >= 4, "not-a-knot spline needs at least 4 points");
        let dx: Vec<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();
        let slope: Vec<f64> = (0..n - 1).map(|i| (y[i + 1] - y[i]) / dx[i]).collect();
        // Tridiagonal system: lower a, diagonal b, upper c, right-hand side d.
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mut c = vec![0.0; n];
        let mut d = vec![0.0; n];
        for i in 1..n - 1 {
            a[i] = dx[i];
            b[i] = 2.0 * (dx[i - 1] + dx[i]);
            c[i] = dx[i - 1];
            d[i] = 3.0 * (dx[i] * slope[i - 1] + dx[i - 1] * slope[i]);
        }
        let d0 = x[2] - x[0];
        b[0] = dx[1];
        c[0] = d0;
        d[0] = ((dx[0] + 2.0 * d0) * dx[1] * slope[0] + dx[0] * dx[0] * slope[1]) / d0;
        let dn = x[n - 1] - x[n - 3];
        b[n - 1] = dx[n - 3];
        a[n - 1] = dn;
        d[n - 1] = (dx[n - 2] * dx[n - 2] * slope[n - 3] + (2.0 * dn + dx[n - 2]) * dx[n - 3] * slope[n - 2]) / dn;
        // Thomas algorithm.
        for i in 1..n {
            let w = a[i] / b[i - 1];
            b[i] -= w * c[i - 1];
            d[i] -= w * d[i - 1];
        }
        let mut s = vec![0.0; n];
        s[n - 1] = d[n - 1] / b[n - 1];
        for i in (0..n - 1).rev() {
            s[i] = (d[i] - c[i] * s[i + 1]) / b[i];
        }
        CubicSpline { x: x.to_vec(), y: y.to_vec(), s }
    }

    pub fn eval(&self, t: f64) -> f64 {
        let n = self.x.len();
        let i = self.x.partition_point(|&v| v <= t).clamp(1, n - 1) - 1;
        let h = self.x[i + 1] - self.x[i];
        let u = (t - self.x[i]) / h;
        let u2 = u * u;
        let u3 = u2 * u;
        (2.0 * u3 - 3.0 * u2 + 1.0) * self.y[i]
            + (u3 - 2.0 * u2 + u) * h * self.s[i]
            + (-2.0 * u3 + 3.0 * u2) * self.y[i + 1]
            + (u3 - u2) * h * self.s[i + 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_legendre_integrates_polynomials() {
        let (x, w) = leggauss(8);
        let s: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(6)).sum();
        assert!((s - 2.0 / 7.0).abs() < 1e-13);
        assert!(x.windows(2).all(|p| p[0] < p[1]));
    }

    #[test]
    fn spline_matches_scipy() {
        // scipy.interpolate.CubicSpline on the same data (not-a-knot), the edge strengths' grid.
        let x: Vec<f64> = (0..403).map(|i| i as f64 * 50.0).collect();
        let y: Vec<f64> = x.iter().map(|t| (t / 3000.0).cos() + 0.1 * (t / 200.0).sin()).collect();
        let s = CubicSpline::new(&x, &y);
        assert!((s.eval(175.0) - 1.0750525975321927).abs() < 1e-14);
        assert!((s.eval(185.0) - 1.0779607188684683).abs() < 1e-14);
    }

    #[test]
    fn spline_is_exact_for_cubics() {
        let x: Vec<f64> = (0..12).map(|i| i as f64 * 0.7 + (i as f64).sqrt()).collect();
        let y: Vec<f64> = x.iter().map(|t| t * t * t - 2.0 * t + 1.0).collect();
        let s = CubicSpline::new(&x, &y);
        for t in [0.3, 2.2, 5.5, 9.9] {
            assert!((s.eval(t) - (t * t * t - 2.0 * t + 1.0)).abs() < 1e-8);
        }
    }

    #[test]
    fn bessel_values() {
        assert!((j1(1.0) - 0.440_050_585_744_933_5).abs() < 1e-12);
        assert!((i0(2.0) - 2.279_585_302_336_067).abs() < 1e-12);
    }
}
