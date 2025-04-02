use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_tags::{BOOLEAN, UNIVERSAL};
use crate::asn1::asn1_write::{get_encoding_type, EncodingType};
use crate::asn1::primitive_encoding::PrimitiveEncoding;
use crate::asn1::{Asn1Encodable, Asn1Write};
use crate::{Error, Result};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::io::Write;

#[derive(Debug)]
pub struct Asn1Boolean {
    value: u8,
}

impl Asn1Boolean {
    pub fn new(value: bool) -> Self {
        Asn1Boolean {
            value: if value { 0xFF } else { 0x00 },
        }
    }
    pub fn with_i32(value: i32) -> Self {
        Asn1Boolean::new(value != 0)
    }
    pub fn is_true(&self) -> bool {
        self.value != 0x00
    }
    fn get_contents(&self, encoding: EncodingType) -> Vec<u8> {
        let mut contents = self.value;
        if self.is_true() && encoding == EncodingType::Der {
            contents = 0xFF;
        }
        vec![contents]
    }
    fn get_encoding_with_type(&self, encode_type: EncodingType) -> impl Asn1Encoding {
        PrimitiveEncoding::new(UNIVERSAL, BOOLEAN, self.get_contents(encode_type))
    }
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        anyhow::ensure!(
            contents.len() == 1,
            Error::invalid_operation("BOOLEAN value should have 1 byte in it")
        );
        Ok(Asn1Boolean::new(contents[0] != 0))
    }
}

impl fmt::Display for Asn1Boolean {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", if self.is_true() { "TRUE" } else { "FALSE" })
    }
}
impl PartialEq for Asn1Boolean {
    fn eq(&self, other: &Self) -> bool {
        self.is_true() == other.is_true()
    }
}
impl Hash for Asn1Boolean {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u8(if self.is_true() { 0xFF } else { 0x00 });
    }
}
impl Asn1Encodable for Asn1Boolean {
    fn encode_to_with_encoding(&self, writer: &mut dyn Write, encoding_str: &str) -> Result<usize> {
        let encoding_type = get_encoding_type(encoding_str);
        let encoding = self.get_encoding_with_type(encoding_type);
        let mut asn1_writer = Asn1Write::create_with_encoding(writer, encoding_str);
        encoding.encode(&mut asn1_writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asn1::asn1_encodable::DER;
    use crate::asn1::Asn1Object;
    #[test]
    fn test_new() {
        {
            let asn1_boolean = Asn1Boolean::new(false);
            assert!(!asn1_boolean.is_true());
        }
        {
            let asn1_boolean = Asn1Boolean::new(true);
            assert!(asn1_boolean.is_true());
        }
    }
    #[test]
    fn test_create_i32() {
        {
            let asn1_boolean = Asn1Boolean::with_i32(1);
            assert!(asn1_boolean.is_true());
        }
        {
            let asn1_boolean = Asn1Boolean::with_i32(0);
            assert!(!asn1_boolean.is_true());
        }
    }
    #[test]
    fn test_display() {
        {
            let asn1_object = Asn1Boolean::new(true);
            assert_eq!("TRUE", asn1_object.to_string());
        }
        {
            let asn1_object = Asn1Boolean::new(false);
            assert_eq!("FALSE", asn1_object.to_string());
        }
    }
    #[test]
    fn test_parse_asn1_object() {
        let buffer = vec![0x01, 0x01, 0xFF];
        let asn1_object = Asn1Object::from_read(&mut buffer.as_slice()).expect("fail");
        assert!(asn1_object.is_boolean());
        let boolean: Asn1Boolean = asn1_object.try_into().unwrap();
        assert!(boolean.is_true());
    }
    #[test]
    fn test_encodable() {
        let result_length = 3;
        let result_buffer = vec![0x01, 0x01, 0xFF];
        let asn1_object = Asn1Boolean::new(true);
        {
            let buffer = asn1_object.get_encoded().expect("fail");
            assert_eq!(result_length, buffer.len());
            assert_eq!(result_buffer, buffer);
        }
        {
            let buffer = asn1_object.get_encoded_with_encoding(DER).expect("fail");
            assert_eq!(result_length, buffer.len());
            assert_eq!(result_buffer, buffer);
        }
        {
            let mut buffer = Vec::<u8>::new();
            let length = asn1_object.encode_to(&mut buffer).expect("fail");
            assert_eq!(result_length, length);
            assert_eq!(result_buffer, buffer);
        }
        {
            let mut buffer = Vec::<u8>::new();
            let length = asn1_object
                .encode_to_with_encoding(&mut buffer, DER)
                .expect("fail");
            assert_eq!(result_length, length);
            assert_eq!(result_buffer, buffer);
        }
    }
}
