use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::{EncodingType};
use crate::asn1::asn1_encoding::Asn1Encoding;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Asn1GraphicString {
    content: String,
}
impl Asn1EncodingInternal for Asn1GraphicString {
    fn get_encoding(&self, _encoding_type: EncodingType) -> Box<dyn Asn1Encoding> {
        todo!()
    }

    fn get_encoding_implicit(&self, _encoding_type: EncodingType, _tag_class: u8, _tag_no: u8) -> Box<dyn Asn1Encoding> {
        todo!()
    }
}