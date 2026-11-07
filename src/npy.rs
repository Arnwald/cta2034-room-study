//! Writing arrays in NumPy's ``.npy`` format, to look at the results with other tools.

use std::io::Write;
use std::path::Path;

pub trait NpyElement: Copy {
    const DESCR: &'static str;
    fn write_le(self, out: &mut Vec<u8>);
}

impl NpyElement for f32 {
    const DESCR: &'static str = "<f4";
    fn write_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl NpyElement for f64 {
    const DESCR: &'static str = "<f8";
    fn write_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl NpyElement for i64 {
    const DESCR: &'static str = "<i8";
    fn write_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

/// Writes ``data`` (row-major) with the given ``shape``.
pub fn write<T: NpyElement>(path: &Path, shape: &[usize], data: &[T]) -> std::io::Result<()> {
    assert_eq!(shape.iter().product::<usize>(), data.len(), "shape does not match the data");
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let shape_str = if dims.len() == 1 { format!("({},)", dims[0]) } else { format!("({})", dims.join(", ")) };
    let mut header = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", T::DESCR, shape_str);
    let total = 10 + header.len() + 1;
    let pad = (64 - total % 64) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len() + data.len() * 8);
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in data {
        v.write_le(&mut out);
    }
    std::fs::File::create(path)?.write_all(&out)
}
