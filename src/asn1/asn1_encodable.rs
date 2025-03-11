use std::io::Write;

use crate::Result;

pub const BER: &str = "BER";
pub const DER: &str = "DER";
pub const DL: &str = "DL";

pub trait Asn1Encodable {
    fn encode_to(&self, writer: &mut dyn Write) -> Result<usize> {
        self.encode_to_with_encoding(writer, BER)
    }
    fn encode_to_with_encoding(&self, writer: &mut dyn Write, encoding: &str) -> Result<usize>;
    fn get_encoded(&self) -> Result<Vec<u8>> {
        self.get_encoded_with_encoding(BER)
    }
    fn get_encoded_with_encoding(&self, encoding: &str) -> Result<Vec<u8>>;
    fn get_der_encoded(&self) -> Option<Vec<u8>> {
        self.get_encoded_with_encoding(DER).ok()
    }
}
