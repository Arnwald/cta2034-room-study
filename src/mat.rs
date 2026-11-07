//! A dense row-major matrix, enough for (directions x frequencies) arrays.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Mat<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

impl<T: Clone + Default> Mat<T> {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat { rows, cols, data: vec![T::default(); rows * cols] }
    }

    pub fn from_rows(rows: Vec<Vec<T>>) -> Self {
        let n = rows.len();
        let cols = if n > 0 { rows[0].len() } else { 0 };
        let mut data = Vec::with_capacity(n * cols);
        for r in rows {
            assert_eq!(r.len(), cols, "ragged rows");
            data.extend(r);
        }
        Mat { rows: n, cols, data }
    }
}

impl<T> Mat<T> {
    pub fn row(&self, i: usize) -> &[T] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        let c = self.cols;
        &mut self.data[i * c..(i + 1) * c]
    }

    pub fn at(&self, i: usize, j: usize) -> &T {
        &self.data[i * self.cols + j]
    }

    pub fn rows_iter(&self) -> impl Iterator<Item = &[T]> {
        self.data.chunks(self.cols.max(1))
    }
}

impl<T: Clone> Mat<T> {
    /// Rows ``start..end`` as a new matrix.
    pub fn slice_rows(&self, start: usize, end: usize) -> Mat<T> {
        Mat { rows: end - start, cols: self.cols, data: self.data[start * self.cols..end * self.cols].to_vec() }
    }

    /// Selected rows as a new matrix.
    pub fn select_rows(&self, idx: &[usize]) -> Mat<T> {
        let mut data = Vec::with_capacity(idx.len() * self.cols);
        for &i in idx {
            data.extend_from_slice(self.row(i));
        }
        Mat { rows: idx.len(), cols: self.cols, data }
    }
}

impl Mat<f32> {
    pub fn column_mean(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.cols];
        for r in self.rows_iter() {
            for (o, v) in out.iter_mut().zip(r) {
                *o += *v as f64;
            }
        }
        out.iter_mut().for_each(|o| *o /= self.rows as f64);
        out
    }
}
