use crate::Result;
use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::primitive_encoding::PrimitiveEncoding;
use crate::asn1::{Asn1String, EncodingType, asn1_tags};
use std::fmt;
use std::fmt::Formatter;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Asn1BmpString {
    content: String,
}
impl Asn1BmpString {
    pub fn new(content: String) -> Self {
        Asn1BmpString { content }
    }
    fn get_contents(&self) -> Vec<u8> {
        self.content.encode_utf16().flat_map(|c| c.to_be_bytes()).collect()
    }
}
impl Asn1String for Asn1BmpString {
    fn to_asn1_string(&self) -> Result<String> {
        Ok(self.content.clone())
    }
}
impl fmt::Display for Asn1BmpString {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.content)
    }
}
impl Asn1EncodingInternal for Asn1BmpString {
    fn get_encoding(&self, encoding_type: EncodingType) -> Box<dyn Asn1Encoding> {
        self.get_encoding_implicit(encoding_type, asn1_tags::UNIVERSAL, asn1_tags::BMP_STRING)
    }

    fn get_encoding_implicit(&self, _encoding_type: EncodingType, tag_class: u8, tag_no: u8) -> Box<dyn Asn1Encoding> {
        Box::new(PrimitiveEncoding::new(tag_class, tag_no, self.get_contents()))
    }
}
