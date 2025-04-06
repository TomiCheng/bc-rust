use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_tags::{INTEGER, UNIVERSAL};
use crate::asn1::asn1_write::{get_encoding_type, EncodingType};
use crate::asn1::primitive_encoding::PrimitiveEncoding;
use crate::asn1::{Asn1Encodable, Asn1Write};
use crate::math::BigInteger;
use crate::{Error, Result};
use anyhow::ensure;
use std::fmt;
use std::io::Write;

pub(crate) const SIGN_EXT_SIGNED: i32 = -1;

#[derive(Clone, Debug)]
pub struct Asn1Integer {
    buffer: Vec<u8>,
    start: usize,
}

impl Asn1Integer {
    pub fn with_i32(value: i32) -> Self {
        let integer = BigInteger::with_i32(value);
        Self {
            start: 0,
            buffer: integer.to_vec(),
        }
    }
    pub fn with_i64(value: i64) -> Self {
        let integer = BigInteger::with_i64(value);
        Self {
            start: 0,
            buffer: integer.to_vec(),
        }
    }
    pub fn with_big_integer(value: &BigInteger) -> Self {
        Self {
            start: 0,
            buffer: value.to_vec(),
        }
    }
    pub fn with_buffer(buffer: Vec<u8>) -> Result<Self> {
        ensure!(
            !is_malformed(&buffer),
            Error::invalid_argument("malformed integer", "buffer")
        );
        let start = sign_bytes_to_skip(&buffer);
        Ok(Asn1Integer { start, buffer })
    }
    pub fn with_buffer_allow_unsafe(buffer: &[u8]) -> Result<Self> {
        ensure!(
            buffer.len() != 0,
            Error::invalid_argument("buffer len is zero", "buffer")
        );
        Ok(Asn1Integer {
            start: sign_bytes_to_skip(&buffer),
            buffer: buffer.to_vec(),
        })
    }
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        Self::with_buffer(contents)
    }
    pub fn get_value(&self) -> BigInteger {
        BigInteger::with_buffer(&self.buffer)
    }

    /// in some cases positive values Get crammed into a space, that's not quite big enough...
    pub fn get_positive_value(&self) -> BigInteger {
        BigInteger::with_sign_buffer(1, &self.buffer).expect("nothing")
    }

    pub fn try_get_i32_value(&self) -> Option<i32> {
        let count = self.buffer.len() - self.start;
        if count > 4 {
            return None;
        }
        Some(get_i32_value(&self.buffer, self.start, SIGN_EXT_SIGNED))
    }

    pub fn has_i32_value(&self, x: i32) -> bool {
        (self.buffer.len() as isize - self.start as isize) <= 4
            && get_i32_value(&self.buffer, self.start, SIGN_EXT_SIGNED) == x
    }

    pub fn try_get_i64_value(&self) -> Option<i64> {
        let count = self.buffer.len() - self.start;
        if count > 8 {
            return None;
        }
        Some(get_i64_value(&self.buffer, self.start, SIGN_EXT_SIGNED))
    }

    pub fn has_i64_value(&self, x: i64) -> bool {
        (self.buffer.len() as isize - self.start as isize) <= 8
            && get_i64_value(&self.buffer, self.start, SIGN_EXT_SIGNED) == x
    }

    fn get_encoding_with_type(&self, _encode_type: EncodingType) -> impl Asn1Encoding {
        PrimitiveEncoding::new(UNIVERSAL, INTEGER, self.buffer.clone())
    }
}

/// Apply the correct validation for an INTEGER primitive following the BER rules.
/// # Arguments
/// * `bytes` - The raw encoding of the integer.
/// # Returns
/// * `true` if the (in)put fails this validation.
pub(crate) fn is_malformed(bytes: &[u8]) -> bool {
    match bytes.len() {
        0 => true,
        1 => false,
        _ => (bytes[0] as i8) == (bytes[1] as i8) >> 7,
    }
}

pub(crate) fn sign_bytes_to_skip(bytes: &[u8]) -> usize {
    let mut pos = 0;
    let last = bytes.len() - 1;
    while pos < last && (bytes[pos] as i8) == (bytes[pos + 1] as i8) >> 7 {
        pos += 1;
    }
    pos
}

pub(crate) fn get_i32_value(bytes: &[u8], start: usize, sign_ext: i32) -> i32 {
    let length = bytes.len();
    let mut pos = isize::max(start as isize, length as isize - 4) as usize;
    let mut val = ((bytes[pos] as i8) as i32) & sign_ext;
    while {
        pos += 1;
        pos
    } < length
    {
        val = (val << 8) | (bytes[pos] as i32);
    }
    val
}

pub(crate) fn get_i64_value(bytes: &[u8], start: usize, sign_ext: i32) -> i64 {
    let length = bytes.len();
    let mut pos = isize::max(start as isize, length as isize - 8) as usize;
    let mut val = ((bytes[pos] as i8) as i64) & (sign_ext as i64);
    while {
        pos += 1;
        pos
    } < length
    {
        val = (val << 8) | (bytes[pos] as i64);
    }
    val
}

// trait
impl Asn1Encodable for Asn1Integer {
    fn encode_to_with_encoding(&self, writer: &mut dyn Write, encoding_str: &str) -> Result<usize> {
        let encoding_type = get_encoding_type(encoding_str);
        let encoding = self.get_encoding_with_type(encoding_type);
        let mut asn1_writer = Asn1Write::create_with_encoding(writer, encoding_str);
        encoding.encode(&mut asn1_writer)
    }
}
impl fmt::Display for Asn1Integer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.get_value().fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use crate::asn1::{Asn1Encodable, Asn1Integer, Asn1Object};
    use crate::asn1::asn1_encodable::DER;
    use crate::math::BigInteger;
    use crate::util::encoders::hex::to_decode_with_str;

    fn check_i32_value(obj: &Asn1Integer, n: i32) {
        let val = obj.get_value();
        assert_eq!(val.i32_value(), n);
        assert_eq!(val.try_get_i32_value(), Some(n));
        assert_eq!(obj.try_get_i32_value(), Some(n));
        assert!(obj.has_i32_value(n));
    }

    fn check_i64_value(obj: &Asn1Integer, n: i64) {
        let val = obj.get_value();
        assert_eq!(val.get_i64_value(), n);
        assert_eq!(val.try_get_i64_value(), Some(n));
        assert_eq!(obj.try_get_i64_value(), Some(n));
        assert!(obj.has_i64_value(n));
    }

    /// Ensure existing single byte behavior.
    #[test]
    fn test_valid_encoding_single_byte() {
        let raw_i32 = vec![0x10];
        let i = Asn1Integer::with_buffer(raw_i32).expect("error");
        check_i32_value(&i, 16);
    }

    #[test]
    fn test_valid_encoding_multi_byte() {
        let raw_i32 = vec![0x10, 0xFF];
        let i = Asn1Integer::with_buffer(raw_i32).expect("error").into();
        check_i32_value(&i, 4351);
    }

    #[test]
    fn test_invalid_encoding_00() {
        let raw_i32 = vec![0x00, 0x10, 0xFF];
        let i = Asn1Integer::with_buffer(raw_i32);
        assert!(i.is_err());
    }

    #[test]
    fn test_invalid_encoding_ff() {
        let raw_i32 = vec![0xFF, 0x81, 0xFF];
        let i = Asn1Integer::with_buffer(raw_i32);
        assert!(i.is_err());
    }

    #[test]
    fn test_invalid_encoding_00_32bits() {
        // Check what would pass loose validation fails outside of loose validation.
        let raw_i32 = vec![0x00, 0x00, 0x00, 0x00, 0x10, 0xFF];
        let i = Asn1Integer::with_buffer(raw_i32);
        assert!(i.is_err());
    }

    #[test]
    fn test_invalid_encoding_ff_32bits() {
        // Check what would pass loose validation fails outside of loose validation.
        let raw_i32 = vec![0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0xFF];
        let i = Asn1Integer::with_buffer(raw_i32);
        assert!(i.is_err());
    }

    #[test]
    fn test_loose_valid_encoding_zero_32b_aligned() {
        let raw_i64 = to_decode_with_str("00000010FF000000").unwrap();
        let i = Asn1Integer::with_buffer_allow_unsafe(&raw_i64)
            .expect("error")
            .into();
        check_i64_value(&i, 72997666816);
    }

    #[test]
    fn test_loose_valid_encoding_ff_32b_aligned() {
        let raw_i64 = to_decode_with_str("FFFFFF10FF000000").unwrap();
        let i = Asn1Integer::with_buffer_allow_unsafe(&raw_i64)
            .expect("error")
            .into();
        check_i64_value(&i, -1026513960960);
    }

    #[test]
    fn test_loose_valid_encoding_ff_32b_aligned_1not0() {
        let raw_i64 = to_decode_with_str("FFFEFF10FF000000").unwrap();
        let i = Asn1Integer::with_buffer_allow_unsafe(&raw_i64)
            .expect("error")
            .into();
        check_i64_value(&i, -282501490671616);
    }

    #[test]
    fn test_loose_valid_encoding_ff_32b_aligned_2not0() {
        let raw_i64 = to_decode_with_str("FFFFFE10FF000000").unwrap();
        let i = Asn1Integer::with_buffer_allow_unsafe(&raw_i64)
            .expect("error")
            .into();
        check_i64_value(&i, -2126025588736);
    }

    #[test]
    fn test_over_sized_encoding() {
        // Should pass as loose validation permits 3 leading 0xFF bytes.
        let der_integer = Asn1Integer::with_buffer_allow_unsafe(
            &to_decode_with_str("FFFFFFFE10FF000000000000").unwrap(),
        )
        .unwrap();
        let big_integer =
            BigInteger::with_buffer(&to_decode_with_str("FFFFFFFE10FF000000000000").unwrap());

        assert_eq!(der_integer.get_value(), big_integer);
    }

    #[test]
    fn test_encode() {
        let result_length = 6;
        let result_buffer = vec![0x02, 0x04, 0x07, 0x5B, 0xCD, 0x15];
        let asn1_integer = Asn1Integer::with_i64(123456789);
        {
            let buffer = asn1_integer.get_encoded().expect("fail");
            assert_eq!(result_length, buffer.len());
            assert_eq!(result_buffer, buffer);
        }
        {
            let buffer = asn1_integer.get_encoded_with_encoding(DER).expect("fail");
            assert_eq!(result_length, buffer.len());
            assert_eq!(result_buffer, buffer);
        }
        {
            let mut buffer = Vec::<u8>::new();
            let length = asn1_integer.encode_to(&mut buffer).expect("fail");
            assert_eq!(result_length, length);
            assert_eq!(result_buffer, buffer);
        }
        {
            let mut buffer = Vec::<u8>::new();
            let length = asn1_integer
                .encode_to_with_encoding(&mut buffer, DER)
                .expect("fail");
            assert_eq!(result_length, length);
            assert_eq!(result_buffer, buffer);
        }
    }

    #[test]
    fn test_parse_asn1_object() {
        let buffer = vec![0x02u8, 0x04, 0x07, 0x5B, 0xCD, 0x15];
        let asn1_object = Asn1Object::from_read(&mut buffer.as_slice()).expect("fail");
        assert!(asn1_object.is_integer());
        let integer: Asn1Integer = asn1_object.try_into().unwrap();
        check_i64_value(&integer, 123456789);
    }
}
