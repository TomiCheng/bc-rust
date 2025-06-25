use std::io::Write;
use crate::asn1::EncodingType;
use crate::Result;

pub trait Asn1Encodable {
    
    fn get_encoded(&self, encoding_type: EncodingType) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        self.encode_to(&mut buffer, encoding_type)?;
        Ok(buffer)
    }
    fn encode_to(&self, writer: &mut dyn Write, encoding_type: EncodingType) -> Result<usize>;
}