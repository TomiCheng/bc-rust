use std::hash::Hash;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_tags::{BOOLEAN, UNIVERSAL};
use crate::asn1::EncodingType;
use crate::asn1::primitive_encoding::PrimitiveEncoding;
use crate::Result;
use std::fmt;
use crate::asn1::asn1_encodable::Asn1EncodingInternal;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Asn1Boolean {
    value: bool,
}

impl Asn1Boolean {
    pub fn new(value: bool) -> Self {
        Asn1Boolean { value }
    }
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        if contents.len() != 1 {
            return Err(crate::error::BcError::with_invalid_operation("BOOLEAN value should have 1 byte in it"));
        }
        let value = contents[0] != 0x00;
        Ok(Asn1Boolean::new(value))
    }
    pub fn is_true(&self) -> bool {
        self.value
    }
    fn get_content(&self, _: EncodingType) -> Vec<u8> {
        if self.value {
            vec![0xffu8]
        } else {
            vec![0x00u8]
        }
    }
}
impl fmt::Display for Asn1Boolean {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.value {
            write!(f, "TRUE")
        } else {
            write!(f, "FALSE")
        }
    }
}
impl Asn1EncodingInternal for Asn1Boolean {
    fn get_encoding(&self, encoding_type: EncodingType) -> Box<dyn Asn1Encoding> {
        Box::new(PrimitiveEncoding::new(UNIVERSAL, BOOLEAN, self.get_content(encoding_type)))
    }
}

#[cfg(test)]
mod tests {
    use crate::asn1::{Asn1Boolean, Asn1Encodable, Asn1Read, EncodingType};

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
    fn test_encodable() {
        let result_length = 3;
        let result_buffer = vec![0x01, 0x01, 0xFF];
        let asn1_object = Asn1Boolean::new(true);
        {
            let buffer = asn1_object.get_encoded(EncodingType::Ber).unwrap();
            assert_eq!(result_length, buffer.len());
            assert_eq!(result_buffer, buffer);
        }
        {
            let buffer = asn1_object.get_encoded(EncodingType::Der).unwrap();
            assert_eq!(result_length, buffer.len());
            assert_eq!(result_buffer, buffer);
        }
        {
            let mut buffer = Vec::<u8>::new();
            let length = asn1_object.encode_to(&mut buffer, EncodingType::Ber).unwrap();
            assert_eq!(result_length, length);
            assert_eq!(result_buffer, buffer);
        }
        {
            let mut buffer = Vec::<u8>::new();
            let length = asn1_object.encode_to(&mut buffer, EncodingType::Der).unwrap();
            assert_eq!(result_length, length);
            assert_eq!(result_buffer, buffer);
        }
    }
    #[test]
    fn test_parse_asn1_object() {
        let buffer = vec![0x01, 0x01, 0xFF];
        let mut slice = buffer.as_slice();
        let mut asn1_read = Asn1Read::new(&mut slice,3);
        let asn1_object = asn1_read.read_object().unwrap().unwrap();

        assert!(asn1_object.is_boolean());
        
        let value = asn1_object.as_boolean().unwrap();
        assert!(value.is_true());
    }
}
