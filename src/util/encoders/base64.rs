//! Base64 encoder and decoder.

use std::io::Write;
use crate::error::{BcError, BcResult};

/// Specifies the Base64 alphabet to use for encoding and decoding.
///
/// Follows [RFC 4648](https://datatracker.ietf.org/doc/html/rfc4648) definitions.
///
/// | Variant | char62 | char63 | padding | RFC |
/// |---------|--------|--------|---------|-----|
/// | `Standard` | `+` | `/` | `=` | RFC 4648 §4 |
/// | `UrlSafe`  | `-` | `_` | `=` | RFC 4648 §5 |
/// | `UrlSafeBc` | `-` | `_` | `.` | bc-csharp `UrlBase64Encoder` |
///
/// ## bc-csharp difference
///
/// bc-csharp's `UrlBase64Encoder` uses `.` as the padding character instead of `=`.
/// This is **not** RFC 4648 compliant but is provided here as [`UrlSafeBc`](Base64Alphabet::UrlSafeBc)
/// for compatibility with bc-csharp encoded data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Base64Alphabet {
    /// RFC 4648 standard Base64: uses `+`, `/` and `=` padding.
    Standard,
    /// RFC 4648 URL-safe Base64: uses `-`, `_` and `=` padding.
    UrlSafe,
    /// bc-csharp compatible URL-safe Base64: uses `-`, `_` and `.` padding.
    /// Non-standard — use only for bc-csharp interoperability.
    UrlSafeBc,
}

struct Alphabet {
    encode_table: [u8; 64],
    decode_table: [u8; 128],
    padding: u8,
}

impl Alphabet {
    const fn new(char62: u8, char63: u8, padding: u8) -> Self {
        let mut encode_table = *b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789  ";
        encode_table[62] = char62;
        encode_table[63] = char63;

        let mut decode_table = [0xFFu8; 128];
        let mut i = 0usize;
        while i < 64 {
            decode_table[encode_table[i] as usize] = i as u8;
            i += 1;
        }

        Self { encode_table, decode_table, padding }
    }
}

const STANDARD: Alphabet = Alphabet::new(b'+', b'/', b'=');
const URL_SAFE: Alphabet = Alphabet::new(b'-', b'_', b'=');
const URL_SAFE_BC: Alphabet = Alphabet::new(b'-', b'_', b'.');

fn is_whitespace(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r')
}

/// Base64 encoder and decoder.
///
/// # Examples
///
/// ```
/// use bc_rust::util::encoders::base64::{Base64Encoder, Base64Alphabet};
///
/// let enc = Base64Encoder::new(Base64Alphabet::Standard);
/// assert_eq!(enc.to_base64_string(b"Hello"), "SGVsbG8=");
/// assert_eq!(enc.decode_str("SGVsbG8=").unwrap(), b"Hello");
///
/// let url_enc = Base64Encoder::new(Base64Alphabet::UrlSafe);
/// assert_eq!(url_enc.to_base64_string(b"Hello"), "SGVsbG8=");
///
/// // bc-csharp compatible URL-safe with '.' padding
/// let bc_enc = Base64Encoder::new(Base64Alphabet::UrlSafeBc);
/// assert_eq!(bc_enc.to_base64_string(b"Hello"), "SGVsbG8.");
/// ```
pub struct Base64Encoder {
    alphabet: &'static Alphabet,
}

impl Base64Encoder {
    /// Creates a new Base64 encoder with the specified alphabet.
    pub fn new(alphabet: Base64Alphabet) -> Self {
        match alphabet {
            Base64Alphabet::Standard => Self { alphabet: &STANDARD },
            Base64Alphabet::UrlSafe => Self { alphabet: &URL_SAFE },
            Base64Alphabet::UrlSafeBc => Self { alphabet: &URL_SAFE_BC },
        }
    }

    /// Encodes `data` as Base64, writing to `out`.
    ///
    /// Returns the number of bytes written.
    ///
    /// # Errors
    ///
    /// Returns [`BcError::IoError`] if writing to `out` fails.
    pub fn encode(&self, data: &[u8], out: &mut impl Write) -> BcResult<usize> {
        let table = &self.alphabet.encode_table;
        let padding = self.alphabet.padding;
        let mut buf = [0u8; 4];
        let mut i = 0;
        let len = data.len();

        while i + 3 <= len {
            let a1 = data[i] as u32;
            let a2 = data[i + 1] as u32;
            let a3 = data[i + 2] as u32;

            buf[0] = table[((a1 >> 2) & 0x3F) as usize];
            buf[1] = table[(((a1 << 4) | (a2 >> 4)) & 0x3F) as usize];
            buf[2] = table[(((a2 << 2) | (a3 >> 6)) & 0x3F) as usize];
            buf[3] = table[(a3 & 0x3F) as usize];

            out.write_all(&buf)?;
            i += 3;
        }

        match len - i {
            1 => {
                let a1 = data[i] as u32;
                buf[0] = table[((a1 >> 2) & 0x3F) as usize];
                buf[1] = table[((a1 << 4) & 0x3F) as usize];
                buf[2] = padding;
                buf[3] = padding;
                out.write_all(&buf)?;
            }
            2 => {
                let a1 = data[i] as u32;
                let a2 = data[i + 1] as u32;
                buf[0] = table[((a1 >> 2) & 0x3F) as usize];
                buf[1] = table[(((a1 << 4) | (a2 >> 4)) & 0x3F) as usize];
                buf[2] = table[((a2 << 2) & 0x3F) as usize];
                buf[3] = padding;
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

    /// Decodes a Base64 string, ignoring whitespace.
    ///
    /// # Errors
    ///
    /// See [`decode`](Self::decode) for error conditions.
    pub fn decode_str(&self, data: &str) -> BcResult<Vec<u8>> {
        let mut out = Vec::new();
        self.decode(data.as_bytes(), &mut out)?;
        Ok(out)
    }

    fn decode_char(&self, c: u8) -> Option<u8> {
        if c < 128 {
            let v = self.alphabet.decode_table[c as usize];
            if v != 0xFF { return Some(v); }
        }
        None
    }

    fn decode_filtered(&self, data: &[u8], out: &mut impl Write) -> BcResult<usize> {
        if data.is_empty() {
            return Ok(0);
        }

        let padding = self.alphabet.padding;
        let end = data.iter().rposition(|&c| c != padding).map_or(0, |i| i + 1);
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

            let b1 = self.decode_char(data[i]).ok_or_else(|| BcError::InvalidArgument {
                param: None,
                msg: format!("invalid base64 character: 0x{:02X}", data[i]),
            })?;
            let b2 = self.decode_char(data[i + 1]).ok_or_else(|| BcError::InvalidArgument {
                param: None,
                msg: format!("invalid base64 character: 0x{:02X}", data[i + 1]),
            })?;

            out.write_all(&[(b1 << 2) | (b2 >> 4)])?;
            count += 1;

            if is_last && padding_count == 2 {
                break;
            }

            let b3 = self.decode_char(data[i + 2]).ok_or_else(|| BcError::InvalidArgument {
                param: None,
                msg: format!("invalid base64 character: 0x{:02X}", data[i + 2]),
            })?;

            out.write_all(&[(b2 << 4) | (b3 >> 2)])?;
            count += 1;

            if is_last && padding_count == 1 {
                break;
            }

            let b4 = self.decode_char(data[i + 3]).ok_or_else(|| BcError::InvalidArgument {
                param: None,
                msg: format!("invalid base64 character: 0x{:02X}", data[i + 3]),
            })?;

            out.write_all(&[(b3 << 6) | b4])?;
            count += 1;

            i += 4;
        }

        Ok(count)
    }
}

impl Default for Base64Encoder {
    fn default() -> Self {
        Self::new(Base64Alphabet::Standard)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_empty() {
        let enc = Base64Encoder::new(Base64Alphabet::Standard);
        assert_eq!(enc.to_base64_string(b""), "");
    }

    #[test]
    fn test_encode_one_byte() {
        let enc = Base64Encoder::new(Base64Alphabet::Standard);
        assert_eq!(enc.to_base64_string(b"M"), "TQ==");
    }

    #[test]
    fn test_encode_two_bytes() {
        let enc = Base64Encoder::new(Base64Alphabet::Standard);
        assert_eq!(enc.to_base64_string(b"Ma"), "TWE=");
    }

    #[test]
    fn test_encode_three_bytes() {
        let enc = Base64Encoder::new(Base64Alphabet::Standard);
        assert_eq!(enc.to_base64_string(b"Man"), "TWFu");
    }

    #[test]
    fn test_encode_hello() {
        let enc = Base64Encoder::new(Base64Alphabet::Standard);
        assert_eq!(enc.to_base64_string(b"Hello"), "SGVsbG8=");
    }

    #[test]
    fn test_encode_url_safe() {
        let enc = Base64Encoder::new(Base64Alphabet::UrlSafe);
        assert_eq!(enc.to_base64_string(b"Hello"), "SGVsbG8=");
    }

    #[test]
    fn test_encode_url_safe_bc() {
        let enc = Base64Encoder::new(Base64Alphabet::UrlSafeBc);
        assert_eq!(enc.to_base64_string(b"Hello"), "SGVsbG8.");
    }

    #[test]
    fn test_decode_empty() {
        let enc = Base64Encoder::new(Base64Alphabet::Standard);
        assert_eq!(enc.decode_str("").unwrap(), b"");
    }

    #[test]
    fn test_decode_one_byte() {
        let enc = Base64Encoder::new(Base64Alphabet::Standard);
        assert_eq!(enc.decode_str("TQ==").unwrap(), b"M");
    }

    #[test]
    fn test_decode_two_bytes() {
        let enc = Base64Encoder::new(Base64Alphabet::Standard);
        assert_eq!(enc.decode_str("TWE=").unwrap(), b"Ma");
    }

    #[test]
    fn test_decode_hello() {
        let enc = Base64Encoder::new(Base64Alphabet::Standard);
        assert_eq!(enc.decode_str("SGVsbG8=").unwrap(), b"Hello");
    }

    #[test]
    fn test_decode_url_safe() {
        let enc = Base64Encoder::new(Base64Alphabet::UrlSafe);
        assert_eq!(enc.decode_str("SGVsbG8=").unwrap(), b"Hello");
    }

    #[test]
    fn test_decode_url_safe_bc() {
        let enc = Base64Encoder::new(Base64Alphabet::UrlSafeBc);
        assert_eq!(enc.decode_str("SGVsbG8.").unwrap(), b"Hello");
    }

    #[test]
    fn test_decode_invalid() {
        let enc = Base64Encoder::new(Base64Alphabet::Standard);
        assert!(enc.decode_str("S!Vs").is_err());
    }

    #[test]
    fn test_roundtrip_standard() {
        let enc = Base64Encoder::new(Base64Alphabet::Standard);
        let original = b"The quick brown fox jumps over the lazy dog";
        let encoded = enc.to_base64_string(original);
        let decoded = enc.decode_str(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_url_safe() {
        let enc = Base64Encoder::new(Base64Alphabet::UrlSafe);
        let original = b"The quick brown fox jumps over the lazy dog";
        let encoded = enc.to_base64_string(original);
        let decoded = enc.decode_str(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_url_safe_bc() {
        let enc = Base64Encoder::new(Base64Alphabet::UrlSafeBc);
        let original = b"The quick brown fox jumps over the lazy dog";
        let encoded = enc.to_base64_string(original);
        let decoded = enc.decode_str(&encoded).unwrap();
        assert_eq!(decoded, original);
    }
}
