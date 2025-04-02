use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_tags::{NULL, UNIVERSAL};
use crate::asn1::asn1_write::{get_encoding_type, EncodingType};
use crate::asn1::primitive_encoding::PrimitiveEncoding;
use crate::asn1::{Asn1Encodable, Asn1Write};
use crate::{Error, Result};
use std::fmt;
use std::io::Write;

/// A Null object.
#[derive(Debug, PartialEq, Hash)]
pub struct Asn1Null;

impl Asn1Null {
    fn new() -> Self {
        Asn1Null {}
    }
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        anyhow::ensure!(
            contents.is_empty(),
            Error::invalid_operation("malformed NULL encoding encountered")
        );
        Ok(Asn1Null::default())
    }
    fn get_encoding_with_type(&self, _encode_type: EncodingType) -> impl Asn1Encoding {
        PrimitiveEncoding::new(UNIVERSAL, NULL, Vec::new())
    }
}

// trait
impl Default for Asn1Null {
    fn default() -> Self {
        Asn1Null::new()
    }
}
impl fmt::Display for Asn1Null {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "NULL")
    }
}

impl Asn1Encodable for Asn1Null {
    fn encode_to_with_encoding(&self, writer: &mut dyn Write, encoding_str: &str) -> Result<usize> {
        let encode_type = get_encoding_type(encoding_str);
        let encoding = self.get_encoding_with_type(encode_type);
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
    fn test_display() {
        let null = Asn1Null::default();
        assert_eq!("NULL".to_string(), null.to_string());
    }
    #[test]
    fn test_parse_asn1_object() {
        let buffer = vec![0x05u8, 0x00];
        let asn1_object = Asn1Object::from_read(&mut buffer.as_slice()).expect("fail");
        assert!(asn1_object.is_null());
    }
    #[test]
    fn test_encodable() {
        let null_buffer = vec![0x05, 0x00];
        let null = Asn1Null::default();
        {
            let buffer = null.get_encoded().expect("fail");
            assert_eq!(2, buffer.len());
            assert_eq!(null_buffer, buffer);
        }
        {
            let buffer = null.get_encoded_with_encoding(DER).expect("fail");
            assert_eq!(2, buffer.len());
            assert_eq!(null_buffer, buffer);
        }
        {
            let mut buffer = Vec::<u8>::new();
            let length = null.encode_to(&mut buffer).expect("fail");
            assert_eq!(2, length);
            assert_eq!(null_buffer, buffer);
        }
        {
            let mut buffer = Vec::<u8>::new();
            let length = null
                .encode_to_with_encoding(&mut buffer, DER)
                .expect("fail");
            assert_eq!(2, length);
            assert_eq!(null_buffer, buffer);
        }
    }
}
