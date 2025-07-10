use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_tags::{OBJECT_IDENTIFIER, UNIVERSAL};
use crate::asn1::asn1_universal_type::Asn1UniversalType;
use crate::asn1::oid_tokenizer::OidTokenizer;
use crate::asn1::primitive_encoding::PrimitiveEncoding;
use crate::asn1::try_from_tagged::TryFromTagged;
use crate::asn1::{Asn1Object, EncodingType, asn1_relative_oid};
use crate::math::BigInteger;
use crate::{BcError, Result};
use std::fmt;
use std::hash::Hash;
use std::sync::OnceLock;

#[derive(Clone, Debug, Eq)]
pub struct Asn1ObjectIdentifier {
    contents: Vec<u8>,
    identifier: OnceLock<String>,
}
impl Asn1ObjectIdentifier {
    pub fn new(contents: Vec<u8>, identifier: OnceLock<String>) -> Self {
        Asn1ObjectIdentifier { contents, identifier }
    }
    pub fn with_str(identifier: &str) -> Result<Self> {
        check_identifier(identifier)?;
        let contents = parse_identifier(identifier)?;
        check_contents_length(contents.len())?;
        Ok(Asn1ObjectIdentifier::new(contents, OnceLock::from(identifier.to_string())))
    }
    pub fn try_parse(identifier: &str) -> Option<Self> {
        if identifier.is_empty() {
            return None;
        }
        if identifier.len() <= MAX_IDENTIFIER_LENGTH && is_valid_identifier(identifier) {
            let contents = parse_identifier(identifier);
            if let Ok(c) = contents {
                return Some(Asn1ObjectIdentifier::new(c, OnceLock::from(identifier.to_string())));
            }
        }
        None
    }
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        check_contents_length(contents.len())?;
        Ok(Asn1ObjectIdentifier::new(contents, OnceLock::new()))
    }
    pub(crate) fn from_asn1_object(asn1_object: Asn1Object) -> Result<Self> {
        if let Some(object_identifier) = asn1_object.as_object_identifier() {
            Ok(object_identifier.clone())
        } else {
            Err(BcError::with_invalid_cast("Expected an octet string for Asn1ObjectIdentifier"))
        }
    }
    pub fn id(&self) -> &String {
        self.identifier.get_or_init(|| parse_contents(self.contents.as_ref()))
    }
    pub fn branch(&self, branch_id: &str) -> Result<Self> {
        asn1_relative_oid::check_identifier(branch_id)?;
        let mut contents = self.contents.clone();
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
        let result = Asn1ObjectIdentifier {
            identifier: OnceLock::new(),
            contents,
        };
        result.identifier.get_or_init(|| format!("{}.{}", root_id, branch_id));
        Ok(result)
    }
    pub fn on(&self, stem: &Self) -> bool {
        let contents = &self.contents;
        let stem_contents = &stem.contents;
        let contents_len = contents.len();
        let stem_contents_len = stem_contents.len();
        contents_len > stem_contents_len && stem_contents == &contents[0..stem_contents_len]
    }
}
impl fmt::Display for Asn1ObjectIdentifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.id())
    }
}
impl PartialEq for Asn1ObjectIdentifier {
    fn eq(&self, other: &Self) -> bool {
        &self.contents == &other.contents
    }
}
impl Hash for Asn1ObjectIdentifier {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.contents.hash(state);
    }
}
impl Asn1EncodingInternal for Asn1ObjectIdentifier {
    fn get_encoding(&self, _: EncodingType) -> Box<dyn Asn1Encoding> {
        Box::new(PrimitiveEncoding::new(UNIVERSAL, OBJECT_IDENTIFIER, self.contents.clone()))
    }
}
impl TryFromTagged for Asn1ObjectIdentifier {
    fn try_from_tagged(tagged: crate::asn1::Asn1TaggedObject, declared_explicit: bool) -> Result<Self> {
        tagged.try_from_base_universal(declared_explicit, Asn1ObjectIdentifierMetadata)
    }
}
struct Asn1ObjectIdentifierMetadata;
impl Asn1UniversalType<Asn1ObjectIdentifier> for Asn1ObjectIdentifierMetadata {
    fn checked_cast(&self, asn1_object: Asn1Object) -> Result<Asn1ObjectIdentifier> {
        asn1_object.try_into()
    }
}

const MAX_CONTENTS_LENGTH: usize = 4096;
const MAX_IDENTIFIER_LENGTH: usize = MAX_CONTENTS_LENGTH * 4 + 1;
pub(crate) fn check_identifier(identifier: &str) -> Result<()> {
    if identifier.len() > MAX_IDENTIFIER_LENGTH {
        return Err(BcError::with_invalid_argument("exceeded relative OID contents length limit"));
    }

    if !is_valid_identifier(identifier) {
        return Err(BcError::with_invalid_argument("string is not a valid relative OID"));
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

    if !asn1_relative_oid::is_valid_identifier(&s[2..]) {
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
    false
}
fn parse_identifier(identifier: &str) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    let mut tokenizer = OidTokenizer::new(identifier);
    let token = tokenizer.next().ok_or(BcError::with_invalid_format("not found first token"))?;

    if !token.chars().all(|c| c >= '0' || c <= '9') {
        return Err(BcError::with_invalid_format("token must be 0 - 9"));
    }

    let first = i64::from_str_radix(token, 10)? * 40;

    let token = tokenizer.next().ok_or(BcError::with_invalid_format("not found second token"))?;

    if !token.chars().all(|c| c >= '0' || c <= '9') {
        return Err(BcError::with_invalid_format("token must be 0 - 9"));
    }

    if token.len() <= 18 {
        asn1_relative_oid::write_field_with_i64(&mut result, first + i64::from_str_radix(token, 10)?)?;
    } else {
        asn1_relative_oid::write_field_with_big_integer(&mut result, &BigInteger::with_string(token)?.add(&BigInteger::with_i64(first)))?;
    }

    for token in tokenizer {
        if !token.chars().all(|c| c >= '0' || c <= '9') {
            return Err(BcError::with_invalid_format("token must be 0 - 9"));
        }
        if token.len() <= 18 {
            asn1_relative_oid::write_field_with_i64(&mut result, i64::from_str_radix(token, 10)?)?;
        } else {
            asn1_relative_oid::write_field_with_big_integer(&mut result, &BigInteger::with_string(token)?)?;
        }
    }
    Ok(result)
}
pub(crate) fn check_contents_length(length: usize) -> Result<()> {
    if length > MAX_IDENTIFIER_LENGTH {
        return Err(BcError::with_invalid_argument("exceeded OID identifier length limit"));
    }
    Ok(())
}
const I64_LIMIT: i64 = (i64::MAX >> 7) - 0x7F;
fn parse_contents(contents: &[u8]) -> String {
    let mut result = String::new();
    let mut value = 0;
    let mut first = true;
    let mut big_value: Option<BigInteger> = None;
    contents.iter().for_each(|b| {
        if value <= I64_LIMIT {
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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::asn1::{Asn1Encodable, Asn1Object};
    use crate::util::encoders::hex::to_decode_with_str;
    use std::io;
    fn recode_check(oid: &str, req: &Vec<u8>) {
        let o = Asn1ObjectIdentifier::with_str(oid).unwrap();
        let mut cursor = io::Cursor::new(req);
        let enc_o = Asn1Object::from_read(&mut cursor).unwrap();

        assert!(enc_o.is_object_identifier());

        let object_identifier: Asn1ObjectIdentifier = enc_o.try_into().unwrap();

        assert_eq!(o, object_identifier);

        let bytes = o.get_der_encoded().unwrap();
        assert_eq!(req, bytes.as_slice());
    }

    fn check_valid(oid: &str) {
        assert!(Asn1ObjectIdentifier::try_parse(oid).is_some());

        let o = Asn1ObjectIdentifier::with_str(oid).unwrap();
        assert_eq!(oid, o.id());
    }

    fn check_invalid(oid: &str) {
        assert!(Asn1ObjectIdentifier::try_parse(oid).is_none());
        assert!(Asn1ObjectIdentifier::with_str(oid).is_err());
    }

    fn check_branch(stem: &str, branch: &str) {
        let mut expected = stem.to_string();
        expected.push('.');
        expected.push_str(branch);

        let binding = Asn1ObjectIdentifier::with_str(stem).unwrap().branch(branch).unwrap();
        let actual = binding.id();
        assert_eq!(&expected, actual);
    }

    fn check_expected(stem: &str, test: &str, expected: bool) {
        let stem_obj = Asn1ObjectIdentifier::with_str(stem).unwrap();
        let test_obj = Asn1ObjectIdentifier::with_str(test).unwrap();
        let actual = test_obj.on(&stem_obj);
        assert_eq!(expected, actual);
    }

    #[test]
    fn record_check_1() {
        let req = to_decode_with_str("0603813403").unwrap();
        recode_check("2.100.3", &req);
    }

    #[test]
    fn record_check_2() {
        let req = to_decode_with_str("06082A36FFFFFFDD6311").unwrap();
        recode_check("1.2.54.34359733987.17", &req);
    }

    #[test]
    fn record_check_value() {
        check_valid("0.1");
        check_valid("1.1.127.32512.8323072.2130706432.545460846592.139637976727552.35747322042253312.9151314442816847872");
        check_valid("1.2.123.12345678901.1.1.1");
        check_valid("2.25.196556539987194312349856245628873852187.1");
        check_valid("0.0");
        check_valid("0.0.1");
        check_valid("0.39");
        check_valid("0.39.1");
        check_valid("1.0");
        check_valid("1.0.1");
        check_valid("1.39");
        check_valid("1.39.1");
        check_valid("2.0");
        check_valid("2.0.1");
        check_valid("2.40");
        check_valid("2.40.1");
    }
    #[test]
    fn check_invalid_1() {
        check_invalid("0");
        check_invalid("1");
        check_invalid("2");
        check_invalid("3.1");
        check_invalid("..1");
        check_invalid("192.168.1.1");
        check_invalid(".123452");
        check_invalid("1.");
        check_invalid("1.345.23.34..234");
        check_invalid("1.345.23.34.234.");
        check_invalid(".12.345.77.234");
        check_invalid(".12.345.77.234.");
        check_invalid("1.2.3.4.A.5");
        check_invalid("1,2");
        check_invalid("0.40");
        check_invalid("0.40.1");
        check_invalid("0.100");
        check_invalid("0.100.1");
        check_invalid("1.40");
        check_invalid("1.40.1");
        check_invalid("1.100");
        check_invalid("1.100.1");
    }

    #[test]
    fn test_branch_check() {
        check_branch("1.1", "2.2");
    }

    #[test]
    fn test_expected() {
        check_expected("1.1", "1.1", false);
        check_expected("1.1", "1.2", false);
        check_expected("1.1", "1.2.1", false);
        check_expected("1.1", "2.1", false);
        check_expected("1.1", "1.11", false);
        check_expected("1.12", "1.1.2", false);
        check_expected("1.1", "1.1.1", true);
        check_expected("1.1", "1.1.2", true);
        check_expected("1.2.3.4.5.6", "1.2.3.4.5.6", false);
        check_expected("1.2.3.4.5.6", "1.2.3.4.5.6.7", true);
        check_expected("1.2.3.4.5.6", "1.2.3.4.5.6.7.8", true);
    }
    #[test]
    fn test_01() {
        check_valid("2.5.4.54");
    }
}
