use std::fmt::Display;
use std::io::Write;
use std::rc::Rc;

use super::asn1_encoding::Asn1Encoding;
use super::asn1_object::{encode_to_with_encoding, get_encoded_with_encoding, Asn1ObjectImpl};
use super::asn1_tags::{NULL, UNIVERSAL};
use super::asn1_write::{get_encoding_type, EncodingType};
use super::primitive_encoding::PrimitiveEncoding;
use super::{Asn1Encodable, Asn1Object};
use crate::{Error, ErrorKind, Result};

/// A Null object.
#[derive(Clone, Debug)]
pub struct DerNullImpl;

impl DerNullImpl {
    pub fn new() -> Self {
        DerNullImpl {}
    }
    fn get_encoding_with_type(&self, _encode_type: EncodingType) -> Box<dyn Asn1Encoding> {
        Box::new(PrimitiveEncoding::new(UNIVERSAL, NULL, std::sync::Arc::new(vec![])))
    }
    pub(crate) fn with_primitive(contents: &[u8]) -> Result<Self> {
        if !contents.is_empty() {
            Err(Error::with_message(
                ErrorKind::InvalidOperation,
                format!("malformed NULL encoding encountered"),
            ))
        } else {
            Ok(DerNullImpl::new())
        }
    }
}

impl Asn1ObjectImpl for DerNullImpl {}
impl Asn1Encodable for DerNullImpl {
    fn get_encoded_with_encoding(&self, encoding_str: &str) -> Result<Vec<u8>> {
        let encoding = self.get_encoding_with_type(get_encoding_type(encoding_str));
        get_encoded_with_encoding(encoding_str, encoding.as_ref())
    }

    fn encode_to_with_encoding(&self, writer: &mut dyn Write, encoding_str: &str) -> Result<usize> {
        let asn1_encoding = self.get_encoding_with_type(get_encoding_type(encoding_str));
        encode_to_with_encoding(writer, encoding_str, asn1_encoding.as_ref())
    }
}
impl Display for DerNullImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "NULL")
    }
}
impl Into<Asn1Object> for DerNullImpl {
    fn into(self) -> Asn1Object {
        Asn1Object::DerNull(self)
    }
}
