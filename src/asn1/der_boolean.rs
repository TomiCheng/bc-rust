use std::fmt::{Display, Formatter};
use std::io::Write;
use std::rc::Rc;

use super::asn1_encoding::Asn1Encoding;
use super::asn1_object::{encode_to_with_encoding, get_encoded_with_encoding, Asn1ObjectImpl};
use super::asn1_tags::{BOOLEAN, UNIVERSAL};
use super::asn1_write::EncodingType;
use super::primitive_encoding::PrimitiveEncoding;
use super::Asn1Encodable;
use crate::asn1::asn1_write::get_encoding_type;
use crate::{Error, ErrorKind, Result};

#[derive(Clone, Debug)]
pub struct DerBooleanImpl {
    value: u8,
}

impl DerBooleanImpl {
    pub fn new(value: bool) -> Self {
        DerBooleanImpl {
            value: if value { 0xFF } else { 0x00 },
        }
    }
    pub fn with_i32(value: i32) -> Self {
        DerBooleanImpl::new(value != 0)
    }
    pub fn is_true(&self) -> bool {
        self.value != 0x00
    }
    fn get_contents(&self, encoding: EncodingType) -> Vec<u8> {
        let mut contents = self.value;
        match encoding {
            EncodingType::Der if self.is_true() => {
                contents = 0xFF;
            }
            _ => {}
        }
        vec![contents]
    }
    fn get_encoding_with_type(&self, encode_type: EncodingType) -> Box<dyn Asn1Encoding> {
        Box::new(PrimitiveEncoding::new(
            UNIVERSAL,
            BOOLEAN,
            std::sync::Arc::new(self.get_contents(encode_type)),
        ))
    }
    pub(crate) fn with_primitive(contents: &[u8]) -> Result<Self> {
        if contents.len() != 1 {
            return Err(Error::with_message(
                ErrorKind::InvalidInput,
                "BOOLEAN value should have 1 byte in it".to_owned(),
            ));
        }
        Ok(DerBooleanImpl::new(contents[0] != 0))
    }
}

impl Asn1ObjectImpl for DerBooleanImpl {}

impl Display for DerBooleanImpl {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", if self.is_true() { "TRUE" } else { "FALSE" })
    }
}

impl PartialEq for DerBooleanImpl {
    fn eq(&self, other: &Self) -> bool {
        self.is_true() == other.is_true()
    }
}

impl Asn1Encodable for DerBooleanImpl {
    fn get_encoded_with_encoding(&self, encoding_str: &str) -> Result<Vec<u8>> {
        let encoding = self.get_encoding_with_type(get_encoding_type(encoding_str));
        get_encoded_with_encoding(encoding_str, encoding.as_ref())
    }

    fn encode_to_with_encoding(&self, writer: &mut dyn Write, encoding_str: &str) -> Result<usize> {
        let asn1_encoding = self.get_encoding_with_type(get_encoding_type(encoding_str));
        encode_to_with_encoding(writer, encoding_str, asn1_encoding.as_ref())
    }
}
