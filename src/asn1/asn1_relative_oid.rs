use std::io::Write;
use std::sync;
use std::fmt;
use std::io;

use super::asn1_object;
use super::asn1_write;
use super::asn1_tags;
use super::asn1_encoding;
use super::primitive_encoding::PrimitiveEncoding;
use super::OidTokenizer;
use crate::math::BigInteger;
use crate::{Error, ErrorKind, Result};

#[derive(Debug)]
pub struct Asn1RelativeOidImpl {
    identifier: sync::OnceLock<String>,
    contents: sync::Arc<Vec<u8>>,
}
impl Asn1RelativeOidImpl {
    fn new(identifier: String, contents: Vec<u8>) -> Asn1RelativeOidImpl {
        let result = Asn1RelativeOidImpl {
            identifier: sync::OnceLock::new(),
            contents: sync::Arc::new(contents),
        };
        let _ = result.identifier.set(identifier);
        result
    }
    pub fn with_str(identifier: &str) -> Result<Asn1RelativeOidImpl> {
        check_identifier(identifier)?;
        let contents = parse_identifier(identifier)?;
        check_contents_length(contents.len())?;
        Ok(Asn1RelativeOidImpl::new(identifier.to_string(), contents))
    }
    pub fn with_vec(contents: Vec<u8>) -> Result<Asn1RelativeOidImpl> {
        check_contents_length(contents.len())?;

        if !is_valid_contents(&contents) {
            return Err(Error::with_message(
                ErrorKind::InvalidInput,
                "invalid relative OID contents".to_owned(),
            ));
        }

        Ok(Asn1RelativeOidImpl {
            identifier: sync::OnceLock::new(),
            contents: sync::Arc::new(contents),
        })

    }


    pub fn id(&self) -> &str {
        self.identifier.get_or_init(|| self.get_id())
    }

    pub fn get_id(&self) -> String {
        self.identifier.get_or_init(|| parse_contents(&self.contents)).to_string()
    }

    pub fn try_from_id(identifier: &str) -> Option<Asn1RelativeOidImpl> {
        if identifier.is_empty() {
            return None;
        }
        if identifier.len() <= MAX_IDENTIFIER_LENGTH && is_valid_identifier(identifier) {
            let contents = parse_identifier(identifier);
            if let Ok(c) = contents {
                return Some(Asn1RelativeOidImpl::new(
                    identifier.to_string(),
                    c,
                ));
            }
        }
        None
    }

    fn get_encoding_with_type(
        &self,
        _encode_type: asn1_write::EncodingType,
    ) -> Box<dyn asn1_encoding::Asn1Encoding> {
        Box::new(PrimitiveEncoding::new(
            asn1_tags::UNIVERSAL,
            asn1_tags::RELATIVE_OID,
            self.contents.clone(),
        ))
    }

    pub fn branch(&self, branch_id: &str) -> Result<Self> {
        check_identifier(branch_id)?;
        let mut contents = self.contents.as_ref().clone();
        if branch_id.len() <= 2 {
            check_contents_length(self.contents.len() + 1)?;
            let mut sub_id = branch_id.chars().nth(0).unwrap() as u32 - '0' as u32;
            if branch_id.len() == 2 {
                sub_id *= 10;
                sub_id += branch_id.chars().nth(1).unwrap() as u32 - '0' as u32;
            }
            contents.push(sub_id as u8);
        } else {
            let branch_contents = parse_identifier(branch_id)?;
            check_contents_length(self.contents.len() + branch_contents.len())?;
            contents.extend(branch_contents);
        }
        let root_id = self.id();
        Ok(Asn1RelativeOidImpl::new(
            format!("{}.{}", root_id, branch_id),
            contents,
        ))
    }
}

// trait
impl asn1_object::Asn1ObjectImpl for Asn1RelativeOidImpl {}
impl fmt::Display for Asn1RelativeOidImpl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id())
    }
}
impl super::Asn1Encodable for Asn1RelativeOidImpl {
    fn get_encoded_with_encoding(&self, encoding_str: &str) -> Result<Vec<u8>> {
        let encoding = self.get_encoding_with_type(asn1_write::get_encoding_type(encoding_str));
        asn1_object::get_encoded_with_encoding(encoding_str, encoding.as_ref())
    }

    fn encode_to_with_encoding(
        &self,
        writer: &mut dyn io::Write,
        encoding_str: &str,
    ) -> Result<usize> {
        let asn1_encoding =
            self.get_encoding_with_type(asn1_write::get_encoding_type(encoding_str));
        asn1_object::encode_to_with_encoding(writer, encoding_str, asn1_encoding.as_ref())
    }
}
impl Into<super::Asn1Object> for Asn1RelativeOidImpl {
    fn into(self) -> super::Asn1Object {
        super::Asn1Object::Asn1RelativeOid(self)
    }
}
impl PartialEq for Asn1RelativeOidImpl {
    fn eq(&self, other: &Self) -> bool {
        self.contents.as_ref() == other.contents.as_ref()
    }
}

// fn 
pub(crate) fn is_valid_identifier(s: &str) -> bool {
    let mut digit_count = 0;

    let mut ch_next: Option<char> = None;
    for ch in s.chars().rev() {
        if ch == '.' {
            if digit_count == 0 || (digit_count > 1 && ch_next == Some('0')) {
                return false;
            }
            digit_count = 0;
        } else if '0' <= ch && ch <= '9' {
            digit_count += 1;
        } else {
            return false;
        }
        ch_next = Some(ch);
    }
    if digit_count == 0 || (digit_count > 1 && ch_next == Some('0')) {
        return false;
    }
    true
}

pub(crate) fn write_field_with_i64(writer: &mut dyn Write, mut value: i64) -> Result<()> {
    let mut result = [0u8; 9];
    let mut pos = 8;
    result[pos] = (value & 0x7F) as u8;
    while value >= (1 << 7) {
        value >>= 7;
        result[{
            pos -= 1;
            pos
        }] = (value | 0x80) as u8;
    }
    writer.write(&result[pos..])?;
    Ok(())
}

pub(crate) fn write_field_with_big_integer(
    writer: &mut dyn Write,
    value: &BigInteger,
) -> Result<()> {
    let byte_count = (value.bit_length() + 6) / 7;
    if byte_count == 0 {
        writer.write(&[0])?;
    } else {
        let mut tmp_value = value.clone();
        let mut tmp = vec![0u8; byte_count];
        for i in (0..byte_count).rev() {
            tmp[i] = (tmp_value.i32_value() | 0x80) as u8;
            tmp_value = tmp_value.shift_right(7);
        }
        tmp[byte_count - 1] &= 0x7F;
        writer.write(&tmp)?;
    }
    Ok(())
}

const MAX_CONTENTS_LENGTH: usize = 4096;
const MAX_IDENTIFIER_LENGTH: usize = MAX_CONTENTS_LENGTH * 4 + 1;

pub(crate) fn check_identifier(identifier: &str) -> Result<()> {
    if identifier.len() > MAX_IDENTIFIER_LENGTH {
        return Err(Error::with_message(
            ErrorKind::InvalidInput,
            "exceeded relative OID contents length limit".to_owned(),
        ));
    }
    if !is_valid_identifier(identifier) {
        return Err(Error::with_message(
            ErrorKind::InvalidInput,
            format!("string {} noa a valid relative OID", identifier),
        ));
    }
    Ok(())
}

pub(crate) fn parse_identifier(s: &str) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    let tokenizer = OidTokenizer::new(s);
    for token in tokenizer {
        if token.len() <= 18 {
            write_field_with_i64(&mut result, i64::from_str_radix(token, 10)?)?;
        } else {
            write_field_with_big_integer(&mut result, &BigInteger::with_string(token)?)?;
        }
    }
    Ok(result)
}

const LONG_LIMIT: i64 = (i64::MAX >> 7) - 0x7F;
pub(crate) fn parse_contents(contents: &[u8]) -> String {
    let mut result = String::new();
    let mut value = 0;
    let mut first = true;
    let mut big_value: Option<BigInteger> = None;
    contents.iter().for_each(|b| {
        if value <= LONG_LIMIT {
            value += (b & 0x7F) as i64;
            if (b & 0x80) == 0 {
                if first {
                    first = false;
                } else {
                    result.push_str(".");
                }
                result.push_str(&value.to_string());
                value = 0;
            } else {
                value <<= 7;
            }
        } else {
            if let Some(v) = &big_value {
                big_value = Some(v.or(&BigInteger::with_i64((b & 0x7F) as i64)));
            } else {
                big_value = Some(BigInteger::with_i64(value));
            }
            let v = big_value.clone().unwrap();
            if (b & 0x80) == 0 {
                if first {
                    first = false;
                } else {
                    result.push('.');
                }
                result.push_str(&v.to_string());
                big_value = None;
                value = 0;
            } else {
                big_value = Some(v.shift_left(7));
            }
        }
    });
    result
}

pub(crate) fn check_contents_length(length: usize) -> Result<()> {
    if length > MAX_CONTENTS_LENGTH {
        return Err(Error::with_message(
            ErrorKind::InvalidInput,
            "exceeded relative OID contents length limit".to_owned(),
        ));
    }
    Ok(())
}

pub(crate) fn is_valid_contents(contents: &[u8]) -> bool {
    if contents.len() < 1 {
        return false;
    }
    let mut sub_id_start = true;
    for i in 0..contents.len() {
        if sub_id_start && contents[i] == 0x80 {
            return false;
        }
        sub_id_start = contents[i] & 0x80 == 0;
    }
    sub_id_start
}