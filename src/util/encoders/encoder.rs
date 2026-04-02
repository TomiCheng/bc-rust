//! Encoder trait for binary-to-text encodings.

use std::io::Write;
use crate::error::BcResult;

/// Encode and decode byte slices (typically from binary to 7-bit ASCII encodings).
///
/// This is the Rust equivalent of bc-csharp's `IEncoder` interface.
///
/// ## bc-csharp difference
///
/// bc-csharp's `IEncoder` has separate overloads for `byte[]` with offset/length
/// and `ReadOnlySpan<byte>`. In Rust, `&[u8]` already represents a slice with
/// bounds, so a single method suffices.
pub trait Encoder {
    /// Encodes `data`, writing the result to `out`.
    ///
    /// Returns the number of bytes written.
    fn encode(&self, data: &[u8], out: &mut dyn Write) -> BcResult<usize>;

    /// Decodes `data`, writing the result to `out`.
    ///
    /// Returns the number of bytes written.
    fn decode(&self, data: &[u8], out: &mut dyn Write) -> BcResult<usize>;

    /// Decodes a string, writing the result to `out`.
    ///
    /// Returns the number of bytes written.
    fn decode_str(&self, data: &str, out: &mut dyn Write) -> BcResult<usize>;
}
