//! Stream utility functions.
//!
//! Port of `Streams.cs` from bc-csharp.
//!
//! Several bc-csharp methods map directly to Rust standard library:
//!
//! | bc-csharp | Rust |
//! |-----------|------|
//! | `Streams.CopyTo` / `PipeAll` | [`std::io::copy`] |
//! | `Streams.ReadAll` | [`Read::read_to_end`] |
//! | `Streams.ReadFully` | [`Read::read_exact`] |

use super::limited_reader::LimitedReader;
use crate::error::BcResult;
use std::io::{self, Read, Write};

/// Default buffer size for stream operations.
pub const DEFAULT_BUFFER_SIZE: usize = 4096;

/// Copies all bytes from `source` to `destination`.
///
/// Prefer [`std::io::copy`] directly for most cases.
///
/// Returns the number of bytes copied.
///
/// # Errors
///
/// Returns [`BcError::IoError`] if reading or writing fails.
///
/// # Examples
///
/// ```
/// use bc_rust::util::io::streams::pipe_all;
///
/// let mut source = b"hello world".as_ref();
/// let mut dest = Vec::new();
/// let n = pipe_all(&mut source, &mut dest).unwrap();
/// assert_eq!(n, 11);
/// assert_eq!(dest, b"hello world");
/// ```
pub fn pipe_all(source: &mut impl Read, destination: &mut impl Write) -> BcResult<u64> {
    Ok(io::copy(source, destination)?)
}

/// Reads and discards all remaining bytes from `source`.
///
/// Equivalent to bc-csharp's `Streams.Drain`.
///
/// # Errors
///
/// Returns [`BcError::IoError`] if reading fails.
///
/// # Examples
///
/// ```
/// use bc_rust::util::io::streams::drain;
///
/// let mut source = b"hello world".as_ref();
/// drain(&mut source).unwrap();
/// ```
pub fn drain(source: &mut impl Read) -> BcResult<()> {
    io::copy(source, &mut io::sink())?;
    Ok(())
}

/// Reads all bytes from `source` into a `Vec<u8>`.
///
/// Prefer [`Read::read_to_end`] directly for most cases.
///
/// # Errors
///
/// Returns [`BcError::IoError`] if reading fails.
///
/// # Examples
///
/// ```
/// use bc_rust::util::io::streams::read_all;
///
/// let mut source = b"hello world".as_ref();
/// let bytes = read_all(&mut source).unwrap();
/// assert_eq!(bytes, b"hello world");
/// ```
pub fn read_all(source: &mut impl Read) -> BcResult<Vec<u8>> {
    let mut buf = Vec::new();
    source.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Reads up to `limit` bytes from `source` into a `Vec<u8>`.
///
/// Silently stops at `limit` even if more bytes are available.
/// Equivalent to bc-csharp's `Streams.ReadAllLimited`.
///
/// # Errors
///
/// Returns [`BcError::IoError`] if reading fails.
///
/// # Examples
///
/// ```
/// use bc_rust::util::io::streams::read_all_limited;
///
/// let mut source = b"hello world".as_ref();
/// let bytes = read_all_limited(&mut source, 5).unwrap();
/// assert_eq!(bytes, b"hello");
/// ```
pub fn read_all_limited(source: &mut impl Read, limit: usize) -> BcResult<Vec<u8>> {
    let mut buf = Vec::new();
    source.take(limit as u64).read_to_end(&mut buf)?;
    Ok(buf)
}

/// Copies bytes from `source` to `destination`, up to `limit` bytes.
///
/// Returns the number of bytes copied.
///
/// # Errors
///
/// - Returns [`BcError::IoError`] if `source` contains more than `limit` bytes.
/// - Returns [`BcError::IoError`] if reading or writing fails.
///
/// # Examples
///
/// ```
/// use bc_rust::util::io::streams::pipe_all_limited;
///
/// let mut source = b"hello".as_ref();
/// let mut dest = Vec::new();
/// let n = pipe_all_limited(&mut source, 10, &mut dest).unwrap();
/// assert_eq!(n, 5);
///
/// let mut source = b"hello world".as_ref();
/// let mut dest = Vec::new();
/// assert!(pipe_all_limited(&mut source, 5, &mut dest).is_err());
/// ```
pub fn pipe_all_limited(
    source: &mut impl Read,
    limit: u64,
    destination: &mut impl Write,
) -> BcResult<u64> {
    let mut limited = LimitedReader::new(source, limit);
    let bytes = io::copy(&mut limited, destination)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipe_all() {
        let mut source = b"hello world".as_ref();
        let mut dest = Vec::new();
        let n = pipe_all(&mut source, &mut dest).unwrap();
        assert_eq!(n, 11);
        assert_eq!(dest, b"hello world");
    }

    #[test]
    fn test_drain() {
        let mut source = b"hello world".as_ref();
        drain(&mut source).unwrap();
        let mut remaining = Vec::new();
        source.read_to_end(&mut remaining).unwrap();
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_read_all() {
        let mut source = b"hello world".as_ref();
        let bytes = read_all(&mut source).unwrap();
        assert_eq!(bytes, b"hello world");
    }

    #[test]
    fn test_read_all_empty() {
        let mut source = b"".as_ref();
        let bytes = read_all(&mut source).unwrap();
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_read_all_limited() {
        let mut source = b"hello world".as_ref();
        let bytes = read_all_limited(&mut source, 5).unwrap();
        assert_eq!(bytes, b"hello");
    }

    #[test]
    fn test_read_all_limited_within_limit() {
        let mut source = b"hello".as_ref();
        let bytes = read_all_limited(&mut source, 100).unwrap();
        assert_eq!(bytes, b"hello");
    }

    #[test]
    fn test_pipe_all_limited_ok() {
        let mut source = b"hello".as_ref();
        let mut dest = Vec::new();
        let n = pipe_all_limited(&mut source, 10, &mut dest).unwrap();
        assert_eq!(n, 5);
        assert_eq!(dest, b"hello");
    }

    #[test]
    fn test_pipe_all_limited_exact() {
        let mut source = b"hello".as_ref();
        let mut dest = Vec::new();
        let n = pipe_all_limited(&mut source, 5, &mut dest).unwrap();
        assert_eq!(n, 5);
    }

    #[test]
    fn test_pipe_all_limited_overflow() {
        let mut source = b"hello world".as_ref();
        let mut dest = Vec::new();
        assert!(pipe_all_limited(&mut source, 5, &mut dest).is_err());
    }
}
