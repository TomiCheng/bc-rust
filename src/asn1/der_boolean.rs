use std::any;
use std::fmt;
use std::io;
use std::sync;

use super::*;
use crate::{BcError, Result};

#[derive(Clone, Debug)]
pub struct DerBoolean {
    value: u8,
}

impl DerBoolean {
    pub fn new(value: bool) -> Self {
        DerBoolean {
            value: if value { 0xFF } else { 0x00 },
        }
    }
    pub fn with_i32(value: i32) -> Self {
        DerBoolean::new(value != 0)
    }
    pub fn is_true(&self) -> bool {
        self.value != 0x00
    }
    fn get_contents(&self, encoding: asn1_write::EncodingType) -> Vec<u8> {
        let mut contents = self.value;
        match encoding {
            asn1_write::EncodingType::Der if self.is_true() => {
                contents = 0xFF;
            }
            _ => {}
        }
        vec![contents]
    }
    fn get_encoding_with_type(
        &self,
        encode_type: asn1_write::EncodingType,
    ) -> Box<dyn asn1_encoding::Asn1Encoding> {
        Box::new(primitive_encoding::PrimitiveEncoding::new(
            asn1_tags::UNIVERSAL,
            asn1_tags::BOOLEAN,
            sync::Arc::new(self.get_contents(encode_type)),
        ))
    }
    pub(crate) fn with_primitive(contents: &[u8]) -> Result<Self> {
        anyhow::ensure!(
            contents.len() == 1,
            BcError::invalid_operation("BOOLEAN value should have 1 byte in it")
        );
        Ok(DerBoolean::new(contents[0] != 0))
    }
}

impl fmt::Display for DerBoolean {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", if self.is_true() { "TRUE" } else { "FALSE" })
    }
}

impl PartialEq for DerBoolean {
    fn eq(&self, other: &Self) -> bool {
        self.is_true() == other.is_true()
    }
}

impl Asn1Encodable for DerBoolean {
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
impl Asn1Object for DerBoolean {
    fn as_any(&self) -> sync::Arc<dyn any::Any> {
        sync::Arc::new(self.clone())
    }
}