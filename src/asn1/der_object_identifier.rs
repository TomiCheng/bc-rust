use std::cell::{Cell, RefCell};
use std::fmt::{Display, Formatter};
use std::hash::Hash;
use std::io::Write;
use std::rc::Rc;

use super::asn1_object::Asn1ObjectImpl;
use super::{asn1_relative_oid, Asn1Encodable, Asn1Object};
use crate::asn1::OidTokenizer;
use crate::math::BigInteger;
use crate::{Error, ErrorKind, Result};

#[derive(Clone)]
pub struct DerObjectIdentifierImpl {
    identifier: RefCell<String>,
    contents: Rc<Vec<u8>>,
}

impl DerObjectIdentifierImpl {
    fn new(identifier: String, contents: Rc<Vec<u8>>) -> Self {
        DerObjectIdentifierImpl {
            identifier: RefCell::new(identifier),
            contents,
        }
    }
    pub fn get_id(&self) -> String {
        let result = parse_contents(&self.contents);
        let mut id = self.identifier.borrow_mut();
        *id = result.clone();
        return result;
    }

    pub fn with_str(identifier: &str) -> Result<Self> {
        check_identifier(identifier)?;

        let contents = parse_identifier(identifier)?;
        check_contents_length(contents.len())?;
        Ok(DerObjectIdentifierImpl::new(
            identifier.to_string(),
            Rc::new(contents),
        ))
    }

    pub fn with_primitive(contents: Vec<u8>) -> Result<Self> {
        check_contents_length(contents.len())?;
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();
        contents.hash(&mut hasher);
        let mut index = hasher.finish();
        index ^= index >> 20;
        index ^= index >> 10;
        index &= 1023;

        //let index = get_hash_code(&contents);

        todo!();
    }
}

impl Asn1ObjectImpl for DerObjectIdentifierImpl {}
impl Display for DerObjectIdentifierImpl {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.get_id())
    }
}
impl Asn1Encodable for DerObjectIdentifierImpl {
    fn encode_to_with_encoding(&self, writer: &mut dyn Write, encoding: &str) -> Result<usize> {
        todo!()
    }

    fn get_encoded_with_encoding(&self, encoding: &str) -> Result<Vec<u8>> {
        todo!()
    }
}
impl Into<Asn1Object> for DerObjectIdentifierImpl {
    fn into(self) -> Asn1Object {
        Asn1Object::DerObjectIdentifier(self)
    }
}

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
    let mut big_value: Option<BigInteger> = None;
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
                big_value = Some(v.or(&BigInteger::with_i64((b & 0x7F) as i64)));
            } else {
                big_value = Some(BigInteger::with_i64(value));
            }
            let mut v = big_value.clone().unwrap();
            if (b & 0x80) == 0 {
                if first {
                    result.push('2');
                    v = v.subtract(&BigInteger::with_i64(80));
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
    if s.len() > MAX_IDENTIFIER_LENGTH {
        return Err(Error::with_message(
            ErrorKind::InvalidInput,
            "exceeded OID contents length limit".to_string(),
        ));
    }
    if !is_valid_identifier(s) {
        return Err(Error::with_message(
            ErrorKind::InvalidInput,
            format!("string {} not a valid OID", s),
        ));
    }
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
    if s.len() == 3 || s.chars().nth(3) != Some('.') {
        return true;
    }
    if s.len() == 4 || s.chars().nth(4) != Some('.') {
        return s.chars().nth(2).unwrap() < '4';
    }
    return false;
}

fn parse_identifier(identifier: &str) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    let mut tokenizer = OidTokenizer::new(identifier);
    let token = tokenizer.next().ok_or(Error::with_message(
        ErrorKind::InvalidFormat,
        "not found first token".to_owned(),
    ))?;
    let first = i64::from_str_radix(token, 10)? * 40;

    let token = tokenizer.next().ok_or(Error::with_message(
        ErrorKind::InvalidFormat,
        "not found second token".to_owned(),
    ))?;
    if token.len() <= 18 {
        asn1_relative_oid::write_field_with_i64(
            &mut result,
            first + i64::from_str_radix(token, 10)?,
        );
    } else {
        asn1_relative_oid::write_field_with_big_integer(
            &mut result,
            &BigInteger::with_string(token)?.add(&BigInteger::with_i64(first)),
        );
    }

    for token in tokenizer {
        if token.len() <= 18 {
            asn1_relative_oid::write_field_with_i64(&mut result, i64::from_str_radix(token, 10)?);
        } else {
            asn1_relative_oid::write_field_with_big_integer(
                &mut result,
                &BigInteger::with_string(token)?,
            );
        }
    }
    Ok(result)
}

pub(crate) fn check_contents_length(length: usize) -> Result<()> {
    if length > MAX_CONTENTS_LENGTH {
        return Err(Error::with_message(
            ErrorKind::InvalidInput,
            "exceeded OID contents length limit".to_string(),
        ));
    }
    Ok(())
}
