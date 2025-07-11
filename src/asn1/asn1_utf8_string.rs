use crate::Result;
use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::primitive_encoding::PrimitiveEncoding;
use crate::asn1::{Asn1String, EncodingType, asn1_tags};
use std::fmt::Display;
use std::hash::Hash;

#[derive(Clone, Debug, Hash, PartialEq)]
pub struct Asn1Utf8String {
    content: String,
}

impl Asn1Utf8String {
    pub fn new(content: String) -> Self {
        Asn1Utf8String { content }
    }
    pub fn with_str(s: &str) -> Self {
        Asn1Utf8String::new(s.to_string())
    }
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        let s = String::from_utf8(contents)?;
        Ok(Self::new(s))
    }
}

impl Display for Asn1Utf8String {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.content)
    }
}
impl Asn1String for Asn1Utf8String {
    fn to_asn1_string(&self) -> Result<String> {
        Ok(self.content.clone())
    }
}
impl From<Asn1Utf8String> for String {
    fn from(value: Asn1Utf8String) -> Self {
        value.content
    }
}
impl Asn1EncodingInternal for Asn1Utf8String {
    fn get_encoding(&self, encoding_type: EncodingType) -> Box<dyn Asn1Encoding> {
        self.get_encoding_implicit(encoding_type, asn1_tags::UNIVERSAL, asn1_tags::UTF8_STRING)
    }
    fn get_encoding_implicit(&self, _encoding_type: EncodingType, tag_class: u8, tag_no: u8) -> Box<dyn Asn1Encoding> {
        Box::new(PrimitiveEncoding::new(
            tag_class,
            tag_no,
            self.content.as_bytes().to_vec(),
        ))
    }
}
