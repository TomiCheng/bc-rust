use crate::Result;
use crate::asn1::{asn1_tags, Asn1String, EncodingType};
use std::fmt;
use std::fmt::Formatter;
use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::primitive_encoding::PrimitiveEncoding;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Asn1BmpString {
    content: String,
}
impl Asn1BmpString {
    pub fn new(content: String) -> Self {
        Asn1BmpString { content }
    }
    fn get_contents(&self) -> Vec<u8> {
        self.content.encode_utf16()
            .flat_map(|c| c.to_be_bytes())
            .collect()
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
    fn get_encoding(&self, _: EncodingType) -> Box<dyn Asn1Encoding> {
        Box::new(PrimitiveEncoding::new(asn1_tags::UNIVERSAL, asn1_tags::BMP_STRING, self.get_contents()))
    }
}
