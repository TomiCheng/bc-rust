use std::any;
use std::fmt;
use std::io;
use std::sync;

use super::*;
use crate::{BcError, Result};

/// A Null object.
#[derive(Clone, Debug)]
pub struct DerNull;

impl DerNull {
    pub fn new() -> Self {
        DerNull {}
    }
    fn get_encoding_with_type(
        &self,
        _encode_type: asn1_write::EncodingType,
    ) -> Box<dyn asn1_encoding::Asn1Encoding> {
        Box::new(primitive_encoding::PrimitiveEncoding::new(
            asn1_tags::UNIVERSAL,
            asn1_tags::NULL,
            std::sync::Arc::new(vec![]),
        ))
    }
    pub(crate) fn with_primitive(contents: &[u8]) -> Result<Self> {
        anyhow::ensure!(
            contents.is_empty(),
            BcError::invalid_operation("malformed NULL encoding encountered")
        );
        Ok(DerNull::new())
    }
}

// trait
impl Asn1Encodable for DerNull {
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
impl fmt::Display for DerNull {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "NULL")
    }
}
impl PartialEq<dyn Asn1Object> for DerNull {
    fn eq(&self, other: &dyn Asn1Object) -> bool {
        let any = other.as_any();
        any.downcast_ref::<DerNull>().is_some()
    }
}
impl Asn1Object for DerNull {
    fn as_any(&self) -> sync::Arc<dyn any::Any> {
        sync::Arc::new(self.clone())
    }
}
impl Default for DerNull {
    fn default() -> Self {
        DerNull::new()
    }
}
