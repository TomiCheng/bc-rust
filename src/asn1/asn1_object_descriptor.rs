use std::hash::Hash;
use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::{asn1_tags, Asn1GraphicString};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Asn1ObjectDescriptor {
    content: Asn1GraphicString
}
impl Asn1EncodingInternal for Asn1ObjectDescriptor {
    fn get_encoding(&self, encoding_type: crate::asn1::EncodingType) -> Box<dyn crate::asn1::asn1_encoding::Asn1Encoding> {
        self.content.get_encoding_implicit(encoding_type, asn1_tags::UNIVERSAL, asn1_tags::OBJECT_DESCRIPTOR)
    }

    fn get_encoding_implicit(&self, encoding_type: crate::asn1::EncodingType, tag_class: u8, tag_no: u8) -> Box<dyn crate::asn1::asn1_encoding::Asn1Encoding> {
        self.content.get_encoding_implicit(encoding_type, tag_class, tag_no)
    }
}