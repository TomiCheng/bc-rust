// use std::rc::Rc;

// use super::asn1_encoding::Asn1Encoding;
// use super::asn1_object::Asn1ObjectInternal;

// use super::asn1_write::EncodingType;
// use super::primitive_encoding::PrimitiveEncoding;
// use super::{Asn1Convertiable, Asn1Encodable, Asn1ObjectImpl};
// use crate::math::BigInteger;
// use crate::{BcError, Result};

pub(crate) const SIGN_EXT_SIGNED: i32 = -1;

// pub struct DerInteger {
//     buffer: Rc<Vec<u8>>,
//     start: usize,
// }

// impl DerInteger {
//     pub fn with_i32(value: i32) -> Self {
//         let integer = BigInteger::with_i32(value);
//         Self {
//             start: 0,
//             buffer: Rc::new(integer.to_vec()),
//         }
//     }
//     pub fn with_i64(value: i64) -> Self {
//         let integer = BigInteger::with_i64(value);
//         Self {
//             start: 0,
//             buffer: Rc::new(integer.to_vec()),
//         }
//     }
//     pub fn with_big_integer(value: &BigInteger) -> Self {
//         Self {
//             start: 0,
//             buffer: Rc::new(value.to_vec()),
//         }
//     }
//     pub fn with_buffer(buffer: &[u8]) -> Result<DerInteger> {
//         if is_malformed(buffer) {
//             return Err(BcError::InvalidInput("malformed integer".to_string()));
//         }
//         Ok(DerInteger {
//             start: sign_bytes_to_skip(buffer),
//             buffer: Rc::new(buffer.to_vec()),
//         })
//     }
//     pub fn with_buffer_allow_unsafe(buffer: &[u8]) -> Result<DerInteger> {
//         if buffer.len() == 0 {
//             return Err(BcError::InvalidInput("buffer len is zero".to_string()));
//         }
//         Ok(DerInteger {
//             start: sign_bytes_to_skip(buffer),
//             buffer: Rc::new(buffer.to_vec()),
//         })
//     }
//     pub(crate) fn with_primitive(contents: &[u8]) -> Result<Self> {
//         Self::with_buffer(contents)
//     }

//     pub fn get_value(&self) -> BigInteger {
//         BigInteger::with_buffer(&self.buffer)
//     }

//     /// in some cases positive values Get crammed into a space, that's not quite big enough...
//     pub fn get_positive_value(&self) -> BigInteger {
//         BigInteger::with_sign_buffer(1, &self.buffer).expect("nothing")
//     }

//     pub fn try_get_i32_value(&self) -> Option<i32> {
//         let count = self.buffer.len() - self.start;
//         if count > 4 {
//             return None;
//         }
//         Some(get_i32_value(&self.buffer, self.start, SIGN_EXT_SIGNED))
//     }

// }

// impl Asn1Convertiable for DerInteger {
//     fn to_asn1_encodable(self: &Rc<Self>) -> Box<dyn Asn1Encodable> {
//         Box::new(Asn1ObjectImpl::new(self.clone()))
//     }
// }

// impl Asn1ObjectInternal for DerInteger {
//     fn get_encoding_with_type(&self, _encoding: &EncodingType) -> Box<dyn Asn1Encoding> {
//         Box::new(PrimitiveEncoding::new(
//             UNIVERSAL,
//             INTEGER,
//             self.buffer.clone(),
//         ))
//     }
// }

// // use super::asn1_tags::{UNIVERSAL, INTEGER};

// // impl Asn1ObjectInternal for DerInteger {
// //     fn get_encoding_with_type(&self, _encoding: &EncodingType) -> Box<dyn Asn1Encoding> {
// //         Box::new(PrimitiveEncoding::new(UNIVERSAL, INTEGER, &self.buffer))
// //     }

// //     fn get_encoding_with_implicit(&self, _encoding: &EncodingType, tag_class: u32, tag_no: u32) -> Box<dyn Asn1Encoding> {
// //         Box::new(PrimitiveEncoding::new(tag_class, tag_no, &self.buffer))
// //     }

// //     fn get_encoding_der(&self) -> Box<dyn DerEncoding> {
// //         Box::new(PrimitiveDerEncoding::new(UNIVERSAL, INTEGER, &self.buffer))
// //     }

// //     fn get_encoding_der_implicit(&self, tag_class: u32, tag_no: u32) -> Box<dyn super::der_encoding::DerEncoding> {
// //         Box::new(PrimitiveDerEncoding::new(tag_class, tag_no, &self.buffer))
// //     }
// // }

// // impl Asn1Object for DerInteger {

// // }

// // impl Asn1Encodable for DerInteger {
// //     fn encode_to(&self, writer: &mut dyn Write) -> Result<usize> {
// //         todo!()
// //     }
// // }

// // impl Asn1Convertiable for DerInteger {
// //     fn to_asn1_object(&self) -> &dyn Asn1Object {
// //         self
// //     }
// //     fn to_any(&self) -> &dyn std::any::Any {
// //         self
// //     }
// // }

use std::fmt::{Display, Formatter};
use std::io::Write;
use std::rc::Rc;

use super::asn1_encoding::Asn1Encoding;
use super::asn1_object::{encode_to_with_encoding, get_encoded_with_encoding, Asn1ObjectImpl};
use super::asn1_tags::{INTEGER, UNIVERSAL};
use super::asn1_write::{get_encoding_type, EncodingType};
use super::primitive_encoding::PrimitiveEncoding;
use super::{Asn1Encodable, Asn1Object};
use crate::math::BigInteger;
use crate::{Error, ErrorKind, Result};

#[derive(Clone, Debug)]
pub struct DerIntegerImpl {
    buffer: Rc<Vec<u8>>,
    start: usize,
}

impl DerIntegerImpl {
    pub fn with_i32(value: i32) -> Self {
        let integer = BigInteger::with_i32(value);
        Self {
            start: 0,
            buffer: Rc::new(integer.to_vec()),
        }
    }
    pub fn with_i64(value: i64) -> Self {
        let integer = BigInteger::with_i64(value);
        Self {
            start: 0,
            buffer: Rc::new(integer.to_vec()),
        }
    }
    pub fn with_big_integer(value: &BigInteger) -> Self {
        Self {
            start: 0,
            buffer: Rc::new(value.to_vec()),
        }
    }
    pub fn with_buffer(buffer: &[u8]) -> Result<DerIntegerImpl> {
        if is_malformed(buffer) {
            return Err(Error::with_message(ErrorKind::InvalidInput,"malformed integer".to_string()));
        }
        Ok(DerIntegerImpl {
            start: sign_bytes_to_skip(buffer),
            buffer: Rc::new(buffer.to_vec()),
        })
    }
    pub fn with_buffer_allow_unsafe(buffer: &[u8]) -> Result<DerIntegerImpl> {
        if buffer.len() == 0 {
            return Err(Error::with_message(ErrorKind::InvalidInput,"buffer len is zero".to_string()));
        }
        Ok(DerIntegerImpl {
            start: sign_bytes_to_skip(buffer),
            buffer: Rc::new(buffer.to_vec()),
        })
    }
    pub(crate) fn with_primitive(contents: &[u8]) -> Result<Self> {
        Self::with_buffer(contents)
    }

    pub fn get_value(&self) -> BigInteger {
        BigInteger::with_buffer(&self.buffer)
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

    fn get_encoding_with_type(&self, _encode_type: EncodingType) -> Box<dyn Asn1Encoding> {
        Box::new(PrimitiveEncoding::new(
            UNIVERSAL,
            INTEGER,
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

impl Into<Asn1Object> for DerIntegerImpl {
    fn into(self) -> Asn1Object {
        Asn1Object::DerInteger(self)
    }
}

impl Asn1ObjectImpl for DerIntegerImpl {}
impl Asn1Encodable for DerIntegerImpl {
    fn get_encoded_with_encoding(&self, encoding_str: &str) -> Result<Vec<u8>> {
        let encoding = self.get_encoding_with_type(get_encoding_type(encoding_str));
        get_encoded_with_encoding(encoding_str, encoding.as_ref())
    }

    fn encode_to_with_encoding(&self, writer: &mut dyn Write, encoding_str: &str) -> Result<usize> {
        let asn1_encoding = self.get_encoding_with_type(get_encoding_type(encoding_str));
        encode_to_with_encoding(writer, encoding_str, asn1_encoding.as_ref())
    }
}
impl Display for DerIntegerImpl {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        self.get_value().fmt(f)
    }
}
