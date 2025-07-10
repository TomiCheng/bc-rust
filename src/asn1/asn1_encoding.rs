use crate::Result;
use crate::asn1::Asn1Write;
pub(crate) trait Asn1Encoding {
    fn encode(&self, writer: &mut Asn1Write) -> Result<usize>;
    fn get_length(&self) -> usize;
}
