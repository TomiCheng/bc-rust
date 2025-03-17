use std::fmt;
use std::io;
use std::sync;
use std::any;

use super::*;
use crate::math;
use crate::{BcError, Result};

pub(crate) const SIGN_EXT_SIGNED: i32 = -1;

#[derive(Clone, Debug)]
pub struct DerInteger {
    buffer: sync::Arc<Vec<u8>>,
    start: usize,
}

impl DerInteger {
    pub fn with_i32(value: i32) -> Self {
        let integer = math::BigInteger::with_i32(value);
        Self {
            start: 0,
            buffer: std::sync::Arc::new(integer.to_vec()),
        }
    }
    pub fn with_i64(value: i64) -> Self {
        let integer = math::BigInteger::with_i64(value);
        Self {
            start: 0,
            buffer: std::sync::Arc::new(integer.to_vec()),
        }
    }
    pub fn with_big_integer(value: &math::BigInteger) -> Self {
        Self {
            start: 0,
            buffer: std::sync::Arc::new(value.to_vec()),
        }
    }
    pub fn with_buffer(buffer: &[u8]) -> Result<DerInteger> {
        anyhow::ensure!(
            !is_malformed(buffer),
            BcError::invalid_argument("malformed integer", "buffer")
        );
        Ok(DerInteger {
            start: sign_bytes_to_skip(buffer),
            buffer: std::sync::Arc::new(buffer.to_vec()),
        })
    }
    pub fn with_buffer_allow_unsafe(buffer: &[u8]) -> Result<DerInteger> {
        anyhow::ensure!(
            buffer.len() != 0,
            BcError::invalid_argument("buffer len is zero", "buffer")
        );
        Ok(DerInteger {
            start: sign_bytes_to_skip(buffer),
            buffer: std::sync::Arc::new(buffer.to_vec()),
        })
    }
    pub(crate) fn with_primitive(contents: &[u8]) -> Result<Self> {
        Self::with_buffer(contents)
    }

    pub fn get_value(&self) -> math::BigInteger {
        math::BigInteger::with_buffer(&self.buffer)
    }

    // /// in some cases positive values Get crammed into a space, that's not quite big enough...
    // pub fn get_positive_value(&self) -> BigInteger {
    //     BigInteger::with_sign_buffer(1, &self.buffer).expect("nothing")
    // }

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

    fn get_encoding_with_type(&self, _encode_type: asn1_write::EncodingType) -> Box<dyn asn1_encoding::Asn1Encoding> {
        Box::new(primitive_encoding::PrimitiveEncoding::new(
            asn1_tags::UNIVERSAL,
            asn1_tags::INTEGER,
            self.buffer.clone(),
        ))
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
    return pos;
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
impl Asn1Encodable for DerInteger {
    fn get_encoded_with_encoding(&self, encoding_str: &str) -> Result<Vec<u8>> {
        let encoding = self.get_encoding_with_type(asn1_write::get_encoding_type(encoding_str));
        asn1_object::get_encoded_with_encoding(encoding_str, encoding.as_ref())
    }

    fn encode_to_with_encoding(&self, writer: &mut dyn io::Write, encoding_str: &str) -> Result<usize> {
        let asn1_encoding = self.get_encoding_with_type(asn1_write::get_encoding_type(encoding_str));
        asn1_object::encode_to_with_encoding(writer, encoding_str, asn1_encoding.as_ref())
    }
}
impl fmt::Display for DerInteger {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.get_value().fmt(f)
    }
}
impl Asn1Object for DerInteger {
    fn as_any(&self) -> sync::Arc<dyn any::Any> {
        sync::Arc::new(self.clone())
    }
}
