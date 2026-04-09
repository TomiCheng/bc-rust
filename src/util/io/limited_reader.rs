//! A [`Read`] wrapper that enforces a byte limit.
//!
//! Port of `LimitedInputStream.cs` from bc-csharp.

use std::io::{self, Read};

/// A [`Read`] wrapper that enforces a maximum number of bytes that can be read.
///
/// Returns an error if the wrapped reader provides more bytes than the limit.
///
/// Equivalent to bc-csharp's `LimitedInputStream`.
pub struct LimitedReader<R: Read> {
    inner: R,
    limit: i64,
}

impl<R: Read> LimitedReader<R> {
    /// Creates a new `LimitedReader` wrapping `inner` with the given `limit`.
    pub fn new(inner: R, limit: u64) -> Self {
        Self {
            inner,
            limit: limit as i64,
        }
    }

    /// Returns the number of bytes remaining before the limit is reached.
    ///
    /// Equivalent to bc-csharp's `LimitedInputStream.CurrentLimit`.
    pub fn current_limit(&self) -> i64 {
        self.limit
    }
}

impl<R: Read> Read for LimitedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        if n > 0 {
            self.limit -= n as i64;
            if self.limit < 0 {
                return Err(io::Error::other("data overflow"));
            }
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_within_limit() {
        let data = b"hello world";
        let mut limited = LimitedReader::new(data.as_ref(), 20);
        let mut buf = Vec::new();
        limited.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"hello world");
        assert_eq!(limited.current_limit(), 9); // 20 - 11
    }

    #[test]
    fn test_read_exact_limit() {
        let data = b"hello";
        let mut limited = LimitedReader::new(data.as_ref(), 5);
        let mut buf = Vec::new();
        limited.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"hello");
        assert_eq!(limited.current_limit(), 0);
    }

    #[test]
    fn test_read_overflow() {
        let data = b"hello world";
        let mut limited = LimitedReader::new(data.as_ref(), 5);
        let mut buf = Vec::new();
        assert!(limited.read_to_end(&mut buf).is_err());
    }
}
