use crate::Result;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::{Asn1Write, EncodingType};
use std::io::Write;

pub(crate) trait Asn1EncodingInternal {
    fn get_encoding(&self, encoding_type: EncodingType) -> Box<dyn Asn1Encoding>;
    fn get_encoding_implicit(&self, encoding_type: EncodingType, tag_class: u8, tag_no: u8) -> Box<dyn Asn1Encoding>;
}

pub trait Asn1Encodable {
    fn get_encoded(&self, encoding_type: EncodingType) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        self.encode_to(&mut buffer, encoding_type)?;
        Ok(buffer)
    }
    fn get_der_encoded(&self) -> Result<Vec<u8>> {
        self.get_encoded(EncodingType::Der)
    }
    fn encode_to(&self, writer: &mut dyn Write, encoding_type: EncodingType) -> Result<usize>;
}

impl<T> Asn1Encodable for T
where
    T: Asn1EncodingInternal,
{
    fn encode_to(&self, writer: &mut dyn Write, encoding_type: EncodingType) -> Result<usize> {
        let mut asn1_writer = Asn1Write::new(writer, encoding_type);
        let asn1_encoding = self.get_encoding(encoding_type);
        asn1_encoding.encode(&mut asn1_writer)
    }
}
