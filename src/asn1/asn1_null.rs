use crate::Result;
use std::fmt;
use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::{asn1_tags, EncodingType};
use crate::asn1::primitive_encoding::PrimitiveEncoding;

/// A Null object.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Asn1Null;

impl Asn1Null {
    pub fn new() -> Self {
        Asn1Null
    }
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        if !contents.is_empty() {
            return Err(crate::BcError::with_invalid_operation("malformed NULL encoding encountered"));
        }
        Ok(Asn1Null)
    }
}
impl fmt::Display for Asn1Null {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NULL")
    }
}
impl Asn1EncodingInternal for Asn1Null {
    fn get_encoding(&self, encoding_type: EncodingType) -> Box<dyn Asn1Encoding> {
        self.get_encoding_implicit(encoding_type, asn1_tags::UNIVERSAL, asn1_tags::NULL)
    }
    fn get_encoding_implicit(&self, _encoding_type: EncodingType, tag_class: u8, tag_no: u8) -> Box<dyn Asn1Encoding> {
        Box::new(PrimitiveEncoding::new(
            tag_class,
            tag_no,
            Vec::new(),
        ))
    }
}