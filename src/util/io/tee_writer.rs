//! A [`Write`] wrapper that copies all written bytes to a secondary [`Write`].
//!
//! Port of `TeeOutputStream.cs` from bc-csharp.

use std::io::{self, Write};

/// A [`Write`] wrapper that copies all written bytes to a secondary [`Write`].
///
/// Every byte written to this writer is also written to the tee writer.
///
/// Equivalent to bc-csharp's `TeeOutputStream`.
///
/// # Examples
///
/// ```
/// use std::io::Write;
/// use bc_rust::util::io::tee_writer::TeeWriter;
///
/// let mut primary = Vec::new();
/// let mut tee_buf = Vec::new();
/// let mut writer = TeeWriter::new(&mut primary, &mut tee_buf);
///
/// writer.write_all(b"hello world").unwrap();
///
/// assert_eq!(primary, b"hello world");
/// assert_eq!(tee_buf, b"hello world");
/// ```
pub struct TeeWriter<W1: Write, W2: Write> {
    inner: W1,
    tee: W2,
}

impl<W1: Write, W2: Write> TeeWriter<W1, W2> {
    /// Creates a new `TeeWriter` that writes to both `inner` and `tee`.
    pub fn new(inner: W1, tee: W2) -> Self {
        Self { inner, tee }
    }
}

impl<W1: Write, W2: Write> Write for TeeWriter<W1, W2> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.tee.write_all(&buf[..n])?;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()?;
        self.tee.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tee_copies_all_bytes() {
        let mut primary = Vec::new();
        let mut tee_buf = Vec::new();
        let mut writer = TeeWriter::new(&mut primary, &mut tee_buf);

        writer.write_all(b"hello world").unwrap();

        assert_eq!(primary, b"hello world");
        assert_eq!(tee_buf, b"hello world");
    }

    #[test]
    fn test_tee_empty() {
        let mut primary = Vec::new();
        let mut tee_buf = Vec::new();
        let mut writer = TeeWriter::new(&mut primary, &mut tee_buf);

        writer.write_all(b"").unwrap();

        assert!(primary.is_empty());
        assert!(tee_buf.is_empty());
    }

    #[test]
    fn test_tee_multiple_writes() {
        let mut primary = Vec::new();
        let mut tee_buf = Vec::new();
        let mut writer = TeeWriter::new(&mut primary, &mut tee_buf);

        writer.write_all(b"hello").unwrap();
        writer.write_all(b" world").unwrap();

        assert_eq!(primary, b"hello world");
        assert_eq!(tee_buf, b"hello world");
    }
}
