//! Base64 encoder and decoder.

use std::io::Write;
use crate::error::{BcError, BcResult};

const ENCODE_TABLE: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

const PADDING: u8 = b'=';

/// Decoding table indexed by ASCII value. `0xFF` indicates an invalid character.
const DECODE_TABLE: [u8; 128] = build_decode_table();

const fn build_decode_table() -> [u8; 128] {
    let mut table = [0xFFu8; 128];
    let mut i = 0usize;
    while i < 64 {
        table[ENCODE_TABLE[i] as usize] = i as u8;
        i += 1;
    }
    table
}

fn is_whitespace(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r')
}

fn decode_char(c: u8) -> Option<u8> {
    if c < 128 {
        let v = DECODE_TABLE[c as usize];
        if v != 0xFF { return Some(v); }
    }
    None
}

/// Base64 encoder and decoder.
///
/// # Examples
///
/// ```
/// use bc_rust::util::encoders::base64::Base64Encoder;
///
/// let enc = Base64Encoder::new();
/// assert_eq!(enc.to_base64_string(b"Hello"), "SGVsbG8=");
/// assert_eq!(enc.decode_str("SGVsbG8=").unwrap(), b"Hello");
/// ```
pub struct Base64Encoder;

impl Base64Encoder {
    /// Creates a new Base64 encoder.
    pub fn new() -> Self {
        Self
    }

    /// Encodes `data` as Base64, writing to `out`.
    ///
    /// Returns the number of bytes written.
    ///
    /// # Errors
    ///
    /// Returns [`BcError::IoError`] if writing to `out` fails.
    pub fn encode(&self, data: &[u8], out: &mut impl Write) -> BcResult<usize> {
        let mut buf = [0u8; 4];
        let mut i = 0;
        let len = data.len();

        while i + 3 <= len {
            let a1 = data[i] as u32;
            let a2 = data[i + 1] as u32;
            let a3 = data[i + 2] as u32;

            buf[0] = ENCODE_TABLE[((a1 >> 2) & 0x3F) as usize];
            buf[1] = ENCODE_TABLE[(((a1 << 4) | (a2 >> 4)) & 0x3F) as usize];
            buf[2] = ENCODE_TABLE[(((a2 << 2) | (a3 >> 6)) & 0x3F) as usize];
            buf[3] = ENCODE_TABLE[(a3 & 0x3F) as usize];

            out.write_all(&buf)?;
            i += 3;
        }

        match len - i {
            1 => {
                let a1 = data[i] as u32;
                buf[0] = ENCODE_TABLE[((a1 >> 2) & 0x3F) as usize];
                buf[1] = ENCODE_TABLE[((a1 << 4) & 0x3F) as usize];
                buf[2] = PADDING;
                buf[3] = PADDING;
                out.write_all(&buf)?;
            }
            2 => {
                let a1 = data[i] as u32;
                let a2 = data[i + 1] as u32;
                buf[0] = ENCODE_TABLE[((a1 >> 2) & 0x3F) as usize];
                buf[1] = ENCODE_TABLE[(((a1 << 4) | (a2 >> 4)) & 0x3F) as usize];
                buf[2] = ENCODE_TABLE[((a2 << 2) & 0x3F) as usize];
                buf[3] = PADDING;
                out.write_all(&buf)?;
            }
            _ => {}
        }

        Ok(len.div_ceil(3) * 4)
    }

    /// Encodes `data` as a Base64 string.
    pub fn to_base64_string(&self, data: &[u8]) -> String {
        let mut out = Vec::with_capacity(data.len().div_ceil(3) * 4);
        self.encode(data, &mut out).expect("Vec<u8> write cannot fail");
        String::from_utf8(out).expect("base64 output is always valid UTF-8")
    }

    /// Decodes Base64 bytes into `out`, ignoring whitespace.
    ///
    /// Returns the number of bytes written.
    ///
    /// # Errors
    ///
    /// - [`BcError::InvalidArgument`] if an invalid Base64 character is encountered.
    /// - [`BcError::IoError`] if writing to `out` fails.
    pub fn decode(&self, data: &[u8], out: &mut impl Write) -> BcResult<usize> {
        let data: Vec<u8> = data.iter().copied().filter(|&c| !is_whitespace(c)).collect();
        self.decode_filtered(&data, out)
    }

    /// Decodes a Base64 string into `out`, ignoring whitespace.
    ///
    /// Returns the number of bytes written.
    ///
    /// # Errors
    ///
    /// See [`decode`](Self::decode) for error conditions.
    pub fn decode_str(&self, data: &str) -> BcResult<Vec<u8>> {
        let mut out = Vec::new();
        self.decode(data.as_bytes(), &mut out)?;
        Ok(out)
    }

    fn decode_filtered(&self, data: &[u8], out: &mut impl Write) -> BcResult<usize> {
        if data.is_empty() {
            return Ok(0);
        }

        // Strip trailing padding
        let end = data.iter().rposition(|&c| c != PADDING).map_or(0, |i| i + 1);
        let padding_count = data.len() - end;

        if !data.len().is_multiple_of(4) {
            return Err(BcError::InvalidArgument {
                param: None,
                msg: "base64 data length must be a multiple of 4".to_string(),
            });
        }

        let mut count = 0;
        let mut i = 0;

        while i + 4 <= data.len() {
            let is_last = i + 4 == data.len();

            let b1 = decode_char(data[i]).ok_or_else(|| BcError::InvalidArgument {
                param: None,
                msg: format!("invalid base64 character: 0x{:02X}", data[i]),
            })?;
            let b2 = decode_char(data[i + 1]).ok_or_else(|| BcError::InvalidArgument {
                param: None,
                msg: format!("invalid base64 character: 0x{:02X}", data[i + 1]),
            })?;

            out.write_all(&[(b1 << 2) | (b2 >> 4)])?;
            count += 1;

            if is_last && padding_count == 2 {
                break;
            }

            let b3 = decode_char(data[i + 2]).ok_or_else(|| BcError::InvalidArgument {
                param: None,
                msg: format!("invalid base64 character: 0x{:02X}", data[i + 2]),
            })?;

            out.write_all(&[((b2 << 4) | (b3 >> 2))])?;
            count += 1;

            if is_last && padding_count == 1 {
                break;
            }

            let b4 = decode_char(data[i + 3]).ok_or_else(|| BcError::InvalidArgument {
                param: None,
                msg: format!("invalid base64 character: 0x{:02X}", data[i + 3]),
            })?;

            out.write_all(&[((b3 << 6) | b4)])?;
            count += 1;

            i += 4;
        }

        Ok(count)
    }
}

impl Default for Base64Encoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_empty() {
        let enc = Base64Encoder::new();
        assert_eq!(enc.to_base64_string(b""), "");
    }

    #[test]
    fn test_encode_one_byte() {
        let enc = Base64Encoder::new();
        assert_eq!(enc.to_base64_string(b"M"), "TQ==");
    }

    #[test]
    fn test_encode_two_bytes() {
        let enc = Base64Encoder::new();
        assert_eq!(enc.to_base64_string(b"Ma"), "TWE=");
    }

    #[test]
    fn test_encode_three_bytes() {
        let enc = Base64Encoder::new();
        assert_eq!(enc.to_base64_string(b"Man"), "TWFu");
    }

    #[test]
    fn test_encode_hello() {
        let enc = Base64Encoder::new();
        assert_eq!(enc.to_base64_string(b"Hello"), "SGVsbG8=");
    }

    #[test]
    fn test_decode_empty() {
        let enc = Base64Encoder::new();
        assert_eq!(enc.decode_str("").unwrap(), b"");
    }

    #[test]
    fn test_decode_one_byte() {
        let enc = Base64Encoder::new();
        assert_eq!(enc.decode_str("TQ==").unwrap(), b"M");
    }

    #[test]
    fn test_decode_two_bytes() {
        let enc = Base64Encoder::new();
        assert_eq!(enc.decode_str("TWE=").unwrap(), b"Ma");
    }

    #[test]
    fn test_decode_hello() {
        let enc = Base64Encoder::new();
        assert_eq!(enc.decode_str("SGVsbG8=").unwrap(), b"Hello");
    }

    #[test]
    fn test_decode_invalid() {
        let enc = Base64Encoder::new();
        assert!(enc.decode_str("S!Vs").is_err());
    }

    #[test]
    fn test_roundtrip() {
        let enc = Base64Encoder::new();
        let original = b"The quick brown fox jumps over the lazy dog";
        let encoded = enc.to_base64_string(original);
        let decoded = enc.decode_str(&encoded).unwrap();
        assert_eq!(decoded, original);
    }
}
