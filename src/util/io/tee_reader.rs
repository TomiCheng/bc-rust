//! A [`Read`] wrapper that copies all read bytes to a secondary [`Write`].
//!
//! Port of `TeeInputStream.cs` from bc-csharp.

use std::io::{self, Read, Write};

/// A [`Read`] wrapper that copies all read bytes to a secondary [`Write`].
///
/// Named after the Unix `tee` command. Every byte read from the inner reader
/// is also written to the tee writer.
///
/// Equivalent to bc-csharp's `TeeInputStream`.
///
/// # Examples
///
/// ```
/// use std::io::Read;
/// use bc_rust::util::io::tee_reader::TeeReader;
///
/// let data = b"hello world";
/// let mut tee_buf = Vec::new();
/// let mut reader = TeeReader::new(data.as_ref(), &mut tee_buf);
///
/// let mut out = Vec::new();
/// reader.read_to_end(&mut out).unwrap();
///
/// assert_eq!(out, b"hello world");
/// assert_eq!(tee_buf, b"hello world"); // copy in tee
/// ```
pub struct TeeReader<R: Read, W: Write> {
    inner: R,
    tee: W,
}

impl<R: Read, W: Write> TeeReader<R, W> {
    /// Creates a new `TeeReader` that reads from `inner` and copies to `tee`.
    pub fn new(inner: R, tee: W) -> Self {
        Self { inner, tee }
    }
}

impl<R: Read, W: Write> Read for TeeReader<R, W> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        if n > 0 {
            self.tee.write_all(&buf[..n])?;
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tee_copies_all_bytes() {
        let data = b"hello world";
        let mut tee_buf = Vec::new();
        let mut reader = TeeReader::new(data.as_ref(), &mut tee_buf);

        let mut out = Vec::new();
        reader.read_to_end(&mut out).unwrap();

        assert_eq!(out, b"hello world");
        assert_eq!(tee_buf, b"hello world");
    }

    #[test]
    fn test_tee_empty() {
        let data = b"";
        let mut tee_buf = Vec::new();
        let mut reader = TeeReader::new(data.as_ref(), &mut tee_buf);

        let mut out = Vec::new();
        reader.read_to_end(&mut out).unwrap();

        assert!(out.is_empty());
        assert!(tee_buf.is_empty());
    }

    #[test]
    fn test_tee_partial_read() {
        let data = b"hello world";
        let mut tee_buf = Vec::new();
        let mut reader = TeeReader::new(data.as_ref(), &mut tee_buf);

        let mut buf = [0u8; 5];
        reader.read_exact(&mut buf).unwrap();

        assert_eq!(&buf, b"hello");
        assert_eq!(tee_buf, b"hello");
    }
}
