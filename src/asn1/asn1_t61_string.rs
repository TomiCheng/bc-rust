use crate::asn1::{asn1_tags, Asn1String, EncodingType};
use std::hash::Hash;
use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::primitive_encoding::PrimitiveEncoding;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Asn1T61String {
    content: String,
}

impl Asn1String for Asn1T61String {
    fn to_asn1_string(&self) -> crate::Result<String> {
        Ok(self.content.clone())
    }
}
impl Asn1EncodingInternal for Asn1T61String {
    fn get_encoding(&self, encoding_type: EncodingType) -> Box<dyn Asn1Encoding> {
        self.get_encoding_implicit(encoding_type, asn1_tags::UNIVERSAL, asn1_tags::T61_STRING)
    }
    fn get_encoding_implicit(&self, _encoding_type: EncodingType, tag_class: u8, tag_no: u8) -> Box<dyn Asn1Encoding> {
        Box::new(PrimitiveEncoding::new(tag_class, tag_no, self.content.clone().into_bytes()))
    }
}