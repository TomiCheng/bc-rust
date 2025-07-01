use std::cell::OnceCell;
use std::io::Write;
use crate::math::BigInteger;
use crate::{BcError, Result};
use crate::asn1::{Asn1Encodable, Asn1Write, EncodingType};
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_tags::{RELATIVE_OID, UNIVERSAL};
use crate::asn1::oid_tokenizer::OidTokenizer;
use crate::asn1::primitive_encoding::PrimitiveEncoding;

#[derive(Debug, Clone)]
pub struct Asn1RelativeOid {
    contents: Vec<u8>,
    identifier: OnceCell<String>
}

impl Asn1RelativeOid {
    fn new(contents: Vec<u8>, identifier: OnceCell<String>) -> Self {
        Asn1RelativeOid {
            contents,
            identifier,
        }
    }
    pub fn with_str(identifier: &str) -> Result<Self> {
        check_identifier(identifier)?;
        let contents = parse_identifier(identifier)?;
        check_contents_length(contents.len())?;
        let identifier = identifier.to_string();
        Ok(Asn1RelativeOid::new(contents,OnceCell::from(identifier)))
    }
    pub fn try_parse_id(identifier: &str) -> Option<Self> {
        if identifier.is_empty() {
            return None;
        }
        if identifier.len() <= MAX_IDENTIFIER_LENGTH && is_valid_identifier(identifier) {
            let contents = parse_identifier(identifier);
            if let Ok(c) = contents {
                return Some(Asn1RelativeOid::new(c, OnceCell::from(identifier.to_string())));
            }
        }
        None
    }
    pub fn branch(&self, branch_id: &str) -> Result<Self> {
        check_identifier(branch_id)?;
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
            let branch_contents = parse_identifier(branch_id)?;
            check_contents_length(self.contents.len() + branch_contents.len())?;
            contents.extend(branch_contents);
        }
        let root_id = self.id();
        Ok(Asn1RelativeOid::new(contents, OnceCell::from(format!("{}.{}", root_id, branch_id))))
    }
    pub fn id(&self) -> &str {
        self.identifier.get_or_init(|| self.get_id())
    }
    pub fn get_id(&self) -> String {
        self.identifier
            .get_or_init(|| parse_contents(&self.contents))
            .to_string()
    }
    fn get_encoding(&self, _: EncodingType) -> impl Asn1Encoding {
        PrimitiveEncoding::new(UNIVERSAL, RELATIVE_OID, self.contents.clone())
    }
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        check_contents_length(contents.len())?;
        Ok(Asn1RelativeOid::new(contents, OnceCell::new()))
    }
}
impl PartialEq for Asn1RelativeOid {
    fn eq(&self, other: &Self) -> bool {
        &self.contents == &other.contents
    }
}
impl Asn1Encodable for Asn1RelativeOid {
    fn encode_to(&self, writer: &mut dyn Write, encoding_type: EncodingType) -> Result<usize> {
        let mut asn1_writer = Asn1Write::new(writer, encoding_type);
        let length = self.get_encoding(encoding_type).encode(&mut asn1_writer)?;
        Ok(length)
    }
}
const MAX_CONTENTS_LENGTH: usize = 4096;
const MAX_IDENTIFIER_LENGTH: usize = MAX_CONTENTS_LENGTH * 4 + 1;
pub(crate) fn check_identifier(identifier: &str) -> Result<()> {

    if identifier.len() > MAX_IDENTIFIER_LENGTH {
        return Err(BcError::with_invalid_argument(
            "exceeded relative OID contents length limit",
        ));
    }

    if !is_valid_identifier(identifier) {
        return Err(BcError::with_invalid_argument(
            "string is not a valid relative OID",
        ));
    }

    Ok(())
}
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
pub(crate) fn write_field_with_big_integer(writer: &mut dyn Write,value: &BigInteger) -> Result<()> {
    let byte_count = (value.bit_length() + 6) / 7;
    if byte_count == 0 {
        writer.write(&[0])?;
    } else {
        let mut tmp_value = value.clone();
        let mut tmp = vec![0u8; byte_count];
        for i in (0..byte_count).rev() {
            tmp[i] = (tmp_value.as_u32() | 0x80) as u8;
            tmp_value = tmp_value.shift_right(7);
        }
        tmp[byte_count - 1] &= 0x7F;
        writer.write(&tmp)?;
    }
    Ok(())
}
pub(crate) fn check_contents_length(length: usize) -> Result<()> {
    if length >= MAX_CONTENTS_LENGTH {
        return Err(BcError::with_invalid_argument(
            "exceeded relative OID contents length limit",
        ));
    }
    Ok(())
}
const I64_LIMIT: i64 = (i64::MAX >> 7)  - 0x7F;
pub(crate) fn parse_contents(contents: &[u8]) -> String {
    let mut result = String::new();
    let mut value = 0;
    let mut first = true;
    let mut big_value: Option<BigInteger> = None;
    contents.iter().for_each(|b| {
        if value <= I64_LIMIT {
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

#[cfg(test)]
mod tests {
    use crate::asn1::{Asn1Encodable, Asn1Object, Asn1RelativeOid};
    use crate::util::encoders::hex::to_decode_with_str;
    use std::io;
    
    #[test]
    fn record_check_1() {
        let req = to_decode_with_str("0D03813403").unwrap();
        recode_check("180.3", &req);
    }
    #[test]
    fn record_check_2() {
        let req = to_decode_with_str("0D082A36FFFFFFDD6311").unwrap();
        recode_check("42.54.34359733987.17", &req);
    }
    
    #[test]
    fn test_check_valid() {
        check_valid("0");
        check_valid("37");
        check_valid("0.1");
        check_valid("1.0");
        check_valid("1.0.2");
        check_valid("1.0.20");
        check_valid("1.0.200");
        check_valid("1.1.127.32512.8323072.2130706432.545460846592.139637976727552.35747322042253312.9151314442816847872");
        check_valid("1.2.123.12345678901.1.1.1");
        check_valid("2.25.196556539987194312349856245628873852187.1");
        check_valid("3.1");
        check_valid("37.196556539987194312349856245628873852187.100");
        check_valid("192.168.1.1");
    }
    
    #[test]
    fn test_check_invalid() {
        check_invalid("00");
        check_invalid("0.01");
        check_invalid("00.1");
        check_invalid("1.00.2");
        check_invalid("1.0.02");
        check_invalid("1.2.00");
        check_invalid(".1");
        check_invalid("..1");
        check_invalid("3..1");
        check_invalid(".123452");
        check_invalid("1.");
        check_invalid("1.345.23.34..234");
        check_invalid("1.345.23.34.234.");
        check_invalid(".12.345.77.234");
        check_invalid(".12.345.77.234.");
        check_invalid("1.2.3.4.A.5");
        check_invalid("1,2");
    }
    
    #[test]
    fn test_branch_check() {
        branch_check("1.1", "2.2");
    }
    
    fn recode_check(oid: &str, req: &Vec<u8>) {
        let o = Asn1RelativeOid::with_str(oid).unwrap();
        let mut cursor = io::Cursor::new(req);
        let enc_o = Asn1Object::from_read(&mut cursor).unwrap();
    
        assert!(enc_o.is_relative_oid());
        let asn1_relative_oid: Asn1RelativeOid = enc_o.try_into().unwrap();
        assert_eq!(o, asn1_relative_oid);
    
        let bytes = o.get_der_encoded().unwrap();
        assert_eq!(req, bytes.as_slice());
    }
    
    fn check_valid(oid: &str) {
        assert!(Asn1RelativeOid::try_parse_id(oid).is_some());
    }
    
    fn check_invalid(oid: &str) {
        assert!(Asn1RelativeOid::try_parse_id(oid).is_none());
        assert!(Asn1RelativeOid::with_str(oid).is_err());
    }
    
    fn branch_check(stem: &str, branch: &str) {
        let expected = format!("{}.{}", stem, branch);
        let actual = Asn1RelativeOid::with_str(stem)
            .unwrap()
            .branch(branch)
            .unwrap()
            .get_id();
        assert_eq!(expected, actual);
    }
}
