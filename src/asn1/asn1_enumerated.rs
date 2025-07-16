use std::hash::{Hash};
use crate::asn1::asn1_encodable::Asn1EncodingInternal;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Asn1Enumerated {}

impl Asn1EncodingInternal for Asn1Enumerated {
    fn get_encoding(&self, _encoding_type: crate::asn1::EncodingType) -> Box<dyn crate::asn1::asn1_encoding::Asn1Encoding> {
        todo!()
    }

    fn get_encoding_implicit(&self, _encoding_type: crate::asn1::EncodingType, _tag_class: u8, _tag_no: u8) -> Box<dyn crate::asn1::asn1_encoding::Asn1Encoding> {
        todo!()
    }
}