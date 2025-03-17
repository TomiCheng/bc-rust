use std::any;
use std::fmt;
use std::io;
use std::sync;

use super::*;
use crate::math;
use crate::{BcError, Result};

#[derive(Clone, Debug)]
pub struct DerObjectIdentifier {
    identifier: sync::OnceLock<String>,
    contents: sync::Arc<Vec<u8>>,
}

impl DerObjectIdentifier {
    fn new(identifier: String, contents: sync::Arc<Vec<u8>>) -> Self {
        let result = DerObjectIdentifier {
            identifier: sync::OnceLock::new(),
            contents,
        };
        result.identifier.get_or_init(|| identifier);
        result
    }
    pub fn with_str(identifier: &str) -> Result<Self> {
        check_identifier(identifier)?;

        let contents = parse_identifier(identifier)?;
        check_contents_length(contents.len())?;
        Ok(DerObjectIdentifier::new(
            identifier.to_string(),
            sync::Arc::new(contents),
        ))
    }

    pub fn with_primitive(contents: Vec<u8>) -> Result<Self> {
        check_contents_length(contents.len())?;
        // todo cache
        Ok(DerObjectIdentifier {
            identifier: sync::OnceLock::new(),
            contents: sync::Arc::new(contents),
        })
    }

    fn get_encoding_with_type(
        &self,
        _encode_type: asn1_write::EncodingType,
    ) -> Box<dyn asn1_encoding::Asn1Encoding> {
        Box::new(primitive_encoding::PrimitiveEncoding::new(
            asn1_tags::UNIVERSAL,
            asn1_tags::OBJECT_IDENTIFIER,
            self.contents.clone(),
        ))
    }

    pub fn try_from_id(identifier: &str) -> Option<Self> {
        if identifier.is_empty() {
            return None;
        }
        if identifier.len() <= MAX_IDENTIFIER_LENGTH && is_valid_identifier(identifier) {
            let contents = parse_identifier(identifier);
            if let Ok(c) = contents {
                return Some(DerObjectIdentifier::new(
                    identifier.to_string(),
                    sync::Arc::new(c),
                ));
            }
        }
        None
    }

    pub fn id(&self) -> &String {
        self.identifier
            .get_or_init(|| parse_contents(self.contents.as_ref()))
    }

    pub fn branch(&self, branch_id: &str) -> Result<Self> {
        asn1_relative_oid::check_identifier(branch_id)?;
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
            let branch_contents = asn1_relative_oid::parse_identifier(branch_id)?;
            check_contents_length(self.contents.len() + branch_contents.len())?;
            contents.extend(branch_contents);
        }
        let root_id = self.id();
        Ok(DerObjectIdentifier::new(
            format!("{}.{}", root_id, branch_id),
            sync::Arc::new(contents),
        ))
    }

    pub fn on(&self, stem: &Self) -> bool {
        let contents = self.contents.as_ref();
        let stem_contents = stem.contents.as_ref();
        let contents_len = contents.len();
        let stem_contents_len = stem_contents.len();
        contents_len > stem_contents_len && stem_contents == &contents[0..stem_contents_len]
    }
}

// Trait
impl fmt::Display for DerObjectIdentifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.id())
    }
}
impl PartialEq for DerObjectIdentifier {
    fn eq(&self, other: &Self) -> bool {
        self.contents.as_ref() == other.contents.as_ref()
    }
}
impl PartialEq<dyn Asn1Object> for DerObjectIdentifier {
    fn eq(&self, other: &dyn Asn1Object) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<DerObjectIdentifier>() {
            return self == other;
        } else {
            return false;
        }
    }
}
impl Asn1Encodable for DerObjectIdentifier {
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
impl Asn1Object for DerObjectIdentifier {
    fn as_any(&self) -> sync::Arc<dyn any::Any> {
        sync::Arc::new(self.clone())
    }
}

// fn

const LONG_LIMIT: i64 = (i64::MAX >> 7) - 0x7F;
/// Implementation limit on the length of the contents octets for an Object Identifier.
/// # Remarks
/// We adopt the same value used by OpenJDK. In theory there is no limit on the length of the contents, or the
/// number of subidentifiers, or the length of individual subidentifiers. In practice, supporting arbitrary
/// lengths can lead to issues, e.g. denial-of-service attacks when attempting to convert a parsed value to its
/// (decimal) string form.
const MAX_CONTENTS_LENGTH: usize = 4096;
const MAX_IDENTIFIER_LENGTH: usize = MAX_CONTENTS_LENGTH * 4 + 1;

fn parse_contents(contents: &[u8]) -> String {
    let mut result = String::new();
    let mut value = 0;
    let mut first = true;
    let mut big_value: Option<math::BigInteger> = None;
    contents.iter().for_each(|b| {
        if value <= LONG_LIMIT {
            value += (b & 0x7F) as i64;
            if (b & 0x80) == 0 {
                if first {
                    if value < 40 {
                        result.push_str("0");
                    } else if value < 80 {
                        result.push_str("1");
                        value -= 40;
                    } else {
                        result.push_str("2");
                        value -= 80;
                    }
                    first = false;
                }
                result.push_str(".");
                result.push_str(&value.to_string());
            } else {
                value <<= 7;
            }
        } else {
            if let Some(v) = &big_value {
                big_value = Some(v.or(&math::BigInteger::with_i64((b & 0x7F) as i64)));
            } else {
                big_value = Some(math::BigInteger::with_i64(value));
            }
            let mut v = big_value.clone().unwrap();
            if (b & 0x80) == 0 {
                if first {
                    result.push('2');
                    v = v.subtract(&math::BigInteger::with_i64(80));
                    first = false;
                }
                result.push('.');
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

fn check_identifier(s: &str) -> Result<()> {
    anyhow::ensure!(
        s.len() <= MAX_IDENTIFIER_LENGTH,
        BcError::invalid_argument("exceeded OID contents length limit", "s")
    );

    anyhow::ensure!(
        is_valid_identifier(s),
        BcError::invalid_argument(&format!("string {} not a valid OID", s), "s")
    );

    Ok(())
}

fn is_valid_identifier(s: &str) -> bool {
    if s.len() < 3 || s.chars().nth(1) != Some('.') {
        return false;
    }

    let first = s.chars().nth(0).unwrap();
    if first < '0' || first > '2' {
        return false;
    }

    if !super::asn1_relative_oid::is_valid_identifier(&s[2..]) {
        return false;
    }

    if first == '2' {
        return true;
    }
    if s.len() == 3 || s.chars().nth(3) == Some('.') {
        return true;
    }
    if s.len() == 4 || s.chars().nth(4) == Some('.') {
        return s.chars().nth(2).unwrap() < '4';
    }
    return false;
}

fn parse_identifier(identifier: &str) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    let mut tokenizer = OidTokenizer::new(identifier);
    let token = tokenizer
        .next()
        .ok_or(BcError::invalid_format("not found first token"))?;
    anyhow::ensure!(
        token.chars().all(|c| c >= '0' || c <= '9'),
        BcError::invalid_format("token must be 0 - 9")
    );
    let first = i64::from_str_radix(token, 10)? * 40;

    let token = tokenizer
        .next()
        .ok_or(BcError::invalid_format("not found second token"))?;
    anyhow::ensure!(
        token.chars().all(|c| c >= '0' || c <= '9'),
        BcError::invalid_format("token must be 0 - 9")
    );
    if token.len() <= 18 {
        asn1_relative_oid::write_field_with_i64(
            &mut result,
            first + i64::from_str_radix(token, 10)?,
        )?;
    } else {
        asn1_relative_oid::write_field_with_big_integer(
            &mut result,
            &math::BigInteger::with_string(token)?.add(&math::BigInteger::with_i64(first)),
        )?;
    }

    for token in tokenizer {
        anyhow::ensure!(
            token.chars().all(|c| c >= '0' || c <= '9'),
            BcError::invalid_format("token must be 0 - 9")
        );
        if token.len() <= 18 {
            asn1_relative_oid::write_field_with_i64(&mut result, i64::from_str_radix(token, 10)?)?;
        } else {
            asn1_relative_oid::write_field_with_big_integer(
                &mut result,
                &math::BigInteger::with_string(token)?,
            )?;
        }
    }
    Ok(result)
}

pub(crate) fn check_contents_length(length: usize) -> Result<()> {
    anyhow::ensure!(
        length <= MAX_CONTENTS_LENGTH,
        BcError::invalid_argument("exceeded OID contents length limit", "length")
    );
    Ok(())
}
