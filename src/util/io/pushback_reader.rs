//! A [`Read`] wrapper with single-byte pushback support.
//!
//! Port of `PushbackStream.cs` from bc-csharp.

use crate::error::BcResult;
use crate::invalid_op;
use std::io::{self, Read};

/// A [`Read`] wrapper that supports pushing back a single byte.
///
/// A pushed-back byte is returned on the next call to [`Read::read`] or
/// [`Read::read_exact`], before reading from the underlying reader.
///
/// Only one byte can be pushed back at a time.
///
/// Equivalent to bc-csharp's `PushbackStream`.
///
/// # Examples
///
/// ```
/// use std::io::Read;
/// use bc_rust::util::io::pushback_reader::PushbackReader;
///
/// let data = b"hello";
/// let mut reader = PushbackReader::new(data.as_ref());
///
/// let mut buf = [0u8; 1];
/// reader.read_exact(&mut buf).unwrap();
/// assert_eq!(buf[0], b'h');
///
/// reader.unread(b'h').unwrap(); // push back 'h'
///
/// let mut buf = [0u8; 5];
/// reader.read_exact(&mut buf).unwrap();
/// assert_eq!(&buf, b"hello"); // 'h' is read again
/// ```
pub struct PushbackReader<R: Read> {
    inner: R,
    pushed_back: Option<u8>,
}

impl<R: Read> PushbackReader<R> {
    /// Creates a new `PushbackReader` wrapping `inner`.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            pushed_back: None,
        }
    }

    /// Pushes back a single byte.
    ///
    /// The byte will be returned on the next read call.
    ///
    /// # Errors
    ///
    /// Returns [`BcError::InvalidArgument`] if a byte has already been pushed back
    /// and not yet consumed.
    pub fn unread(&mut self, b: u8) -> BcResult<()> {
        if self.pushed_back.is_some() {
            return invalid_op!("can only push back one byte");
        }
        self.pushed_back = Some(b);
        Ok(())
    }
}

impl<R: Read> Read for PushbackReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        if let Some(b) = self.pushed_back.take() {
            buf[0] = b;
            return Ok(1);
        }
        self.inner.read(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_without_pushback() {
        let mut reader = PushbackReader::new(b"hello".as_ref());
        let mut buf = [0u8; 5];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"hello");
    }

    #[test]
    fn test_unread_and_read() {
        let mut reader = PushbackReader::new(b"ello".as_ref());
        reader.unread(b'h').unwrap();
        let mut buf = [0u8; 5];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"hello");
    }

    #[test]
    fn test_unread_twice_fails() {
        let mut reader = PushbackReader::new(b"hello".as_ref());
        reader.unread(b'x').unwrap();
        assert!(reader.unread(b'y').is_err());
    }

    #[test]
    fn test_unread_consumed_after_read() {
        let mut reader = PushbackReader::new(b"bc".as_ref());
        reader.unread(b'a').unwrap();

        let mut buf = [0u8; 1];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(buf[0], b'a');

        // pushed_back is consumed, can push again
        reader.unread(b'x').unwrap();
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(buf[0], b'x');
    }

    #[test]
    fn test_read_empty_buf() {
        let mut reader = PushbackReader::new(b"hello".as_ref());
        reader.unread(b'x').unwrap();
        let mut buf = [];
        assert_eq!(reader.read(&mut buf).unwrap(), 0);
        // pushed_back still present
        let mut buf = [0u8; 1];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(buf[0], b'x');
    }
}
