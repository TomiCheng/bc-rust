//! Hex encoder and decoder.

use crate::error::{BcError, BcResult};
use crate::util::encoders::encoder::Encoder;
use std::io::Write;

const ENCODE_TABLE_LOWER: &[u8; 16] = b"0123456789abcdef";
const ENCODE_TABLE_UPPER: &[u8; 16] = b"0123456789ABCDEF";

/// Decoding table indexed by ASCII value. `0xFF` indicates an invalid character.
const DECODE_TABLE: [u8; 128] = build_decode_table();

const fn build_decode_table() -> [u8; 128] {
    let mut table = [0xFFu8; 128];
    let mut i = 0u8;
    while i < 10 {
        table[(b'0' + i) as usize] = i;
        i += 1;
    }
    let mut i = 0u8;
    while i < 6 {
        table[(b'a' + i) as usize] = 10 + i;
        table[(b'A' + i) as usize] = 10 + i;
        i += 1;
    }
    table
}

fn is_whitespace(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r')
}

fn decode_nibble(c: u8) -> Option<u8> {
    if c < 128 {
        let v = DECODE_TABLE[c as usize];
        if v != 0xFF {
            return Some(v);
        }
    }
    None
}

/// Specifies whether hex output should be lowercase or uppercase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HexCase {
    Lower,
    Upper,
}

/// Hex encoder and decoder.
///
/// # Examples
///
/// ```
/// use bc_rust::util::encoders::hex::{HexEncoder, HexCase};
///
/// let lower = HexEncoder::new(HexCase::Lower);
/// assert_eq!(lower.to_hex_string(&[0xDE, 0xAD]), "dead");
///
/// let upper = HexEncoder::new(HexCase::Upper);
/// assert_eq!(upper.to_hex_string(&[0xDE, 0xAD]), "DEAD");
/// ```
pub struct HexEncoder {
    encode_table: &'static [u8; 16],
}

impl HexEncoder {
    /// Creates a hex encoder with the specified case.
    pub fn new(case: HexCase) -> Self {
        match case {
            HexCase::Lower => Self {
                encode_table: ENCODE_TABLE_LOWER,
            },
            HexCase::Upper => Self {
                encode_table: ENCODE_TABLE_UPPER,
            },
        }
    }

    /// Encodes `data` as hex, writing to `out`.
    ///
    /// Returns the number of bytes written.
    ///
    /// # Errors
    ///
    /// Returns [`BcError::IoError`] if writing to `out` fails.
    pub fn encode(&self, data: &[u8], out: &mut (impl Write + ?Sized)) -> BcResult<usize> {
        let mut buf = [0u8; 2];
        for &b in data {
            buf[0] = self.encode_table[(b >> 4) as usize];
            buf[1] = self.encode_table[(b & 0xF) as usize];
            out.write_all(&buf)?;
        }
        Ok(data.len() * 2)
    }

    /// Encodes `data` as a hex string.
    pub fn to_hex_string(&self, data: &[u8]) -> String {
        let mut out = Vec::with_capacity(data.len() * 2);
        self.encode(data, &mut out)
            .expect("Vec<u8> write cannot fail");
        String::from_utf8(out).expect("hex output is always valid UTF-8")
    }

    /// Decodes hex bytes into `out`, ignoring whitespace.
    ///
    /// Returns the number of bytes written.
    ///
    /// # Errors
    ///
    /// - [`BcError::InvalidArgument`] if the data length (after stripping whitespace) is odd.
    /// - [`BcError::InvalidArgument`] if a non-hex character is encountered.
    /// - [`BcError::IoError`] if writing to `out` fails.
    pub fn decode(&self, data: &[u8], out: &mut (impl Write + ?Sized)) -> BcResult<usize> {
        let data: Vec<u8> = data
            .iter()
            .copied()
            .filter(|&c| !is_whitespace(c))
            .collect();
        if !data.len().is_multiple_of(2) {
            return Err(BcError::InvalidArgument {
                param: None,
                msg: "hex data must have an even number of characters".to_string(),
            });
        }
        let mut count = 0;
        for chunk in data.chunks(2) {
            let hi = decode_nibble(chunk[0]).ok_or_else(|| BcError::InvalidArgument {
                param: None,
                msg: format!("invalid hex character: 0x{:02X}", chunk[0]),
            })?;
            let lo = decode_nibble(chunk[1]).ok_or_else(|| BcError::InvalidArgument {
                param: None,
                msg: format!("invalid hex character: 0x{:02X}", chunk[1]),
            })?;
            out.write_all(&[(hi << 4) | lo])?;
            count += 1;
        }
        Ok(count)
    }

    /// Decodes a hex string into `out`, ignoring whitespace.
    ///
    /// Returns the number of bytes written.
    ///
    /// # Errors
    ///
    /// See [`decode`](Self::decode) for error conditions.
    pub fn decode_str(&self, data: &str, out: &mut (impl Write + ?Sized)) -> BcResult<usize> {
        self.decode(data.as_bytes(), out)
    }

    /// Decodes a hex string strictly — no whitespace allowed.
    ///
    /// # Errors
    ///
    /// - [`BcError::InvalidArgument`] if the string length is odd.
    /// - [`BcError::InvalidArgument`] if any character is not a valid hex digit (whitespace included).
    ///
    /// # Examples
    ///
    /// ```
    /// use bc_rust::util::encoders::hex::{HexEncoder, HexCase};
    /// let encoder = HexEncoder::new(HexCase::Lower);
    /// assert_eq!(encoder.decode_strict("deadbeef").unwrap(), vec![0xDE, 0xAD, 0xBE, 0xEF]);
    /// assert!(encoder.decode_strict("dead beef").is_err());
    /// ```
    pub fn decode_strict(&self, data: &str) -> BcResult<Vec<u8>> {
        let bytes = data.as_bytes();
        if !bytes.len().is_multiple_of(2) {
            return Err(BcError::InvalidArgument {
                param: None,
                msg: "hex data must have an even number of characters".to_string(),
            });
        }
        let mut out = Vec::with_capacity(bytes.len() / 2);
        for chunk in bytes.chunks(2) {
            let hi = decode_nibble(chunk[0]).ok_or_else(|| BcError::InvalidArgument {
                param: None,
                msg: format!("invalid hex character: 0x{:02X}", chunk[0]),
            })?;
            let lo = decode_nibble(chunk[1]).ok_or_else(|| BcError::InvalidArgument {
                param: None,
                msg: format!("invalid hex character: 0x{:02X}", chunk[1]),
            })?;
            out.push((hi << 4) | lo);
        }
        Ok(out)
    }
}

impl Default for HexEncoder {
    fn default() -> Self {
        Self::new(HexCase::Lower)
    }
}

impl Encoder for HexEncoder {
    fn encode(&self, data: &[u8], out: &mut dyn Write) -> BcResult<usize> {
        self.encode(data, out)
    }

    fn decode(&self, data: &[u8], out: &mut dyn Write) -> BcResult<usize> {
        self.decode(data, out)
    }

    fn decode_str(&self, data: &str, out: &mut dyn Write) -> BcResult<usize> {
        self.decode_str(data, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_hex_string_lower() {
        let enc = HexEncoder::new(HexCase::Lower);
        assert_eq!(enc.to_hex_string(&[]), "");
        assert_eq!(enc.to_hex_string(&[0x00]), "00");
        assert_eq!(enc.to_hex_string(&[0xDE, 0xAD, 0xBE, 0xEF]), "deadbeef");
    }

    #[test]
    fn test_to_hex_string_upper() {
        let enc = HexEncoder::new(HexCase::Upper);
        assert_eq!(enc.to_hex_string(&[0xDE, 0xAD, 0xBE, 0xEF]), "DEADBEEF");
    }

    #[test]
    fn test_decode_str() {
        let enc = HexEncoder::new(HexCase::Lower);
        let mut out = Vec::new();
        enc.decode_str("deadbeef", &mut out).unwrap();
        assert_eq!(out, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_decode_str_with_whitespace() {
        let enc = HexEncoder::new(HexCase::Lower);
        let mut out = Vec::new();
        enc.decode_str("de ad\nbe\tef", &mut out).unwrap();
        assert_eq!(out, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_decode_strict() {
        let enc = HexEncoder::new(HexCase::Lower);
        assert_eq!(
            enc.decode_strict("deadbeef").unwrap(),
            vec![0xDE, 0xAD, 0xBE, 0xEF]
        );
        assert!(enc.decode_strict("dead beef").is_err());
        assert!(enc.decode_strict("xyz").is_err());
    }

    #[test]
    fn test_roundtrip() {
        let enc = HexEncoder::new(HexCase::Lower);
        let original = vec![0x00, 0x01, 0x7F, 0x80, 0xFF];
        let encoded = enc.to_hex_string(&original);
        let decoded = enc.decode_strict(&encoded).unwrap();
        assert_eq!(original, decoded);
    }
}
