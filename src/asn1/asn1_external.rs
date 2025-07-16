use std::hash::{Hash};
use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::EncodingType;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Asn1External {}

impl Asn1EncodingInternal for Asn1External {
    fn get_encoding(&self, encoding_type: EncodingType) -> Box<dyn Asn1Encoding> {
        todo!()
    }

    fn get_encoding_implicit(&self, encoding_type: EncodingType, tag_class: u8, tag_no: u8) -> Box<dyn Asn1Encoding> {
        todo!()
    }
}