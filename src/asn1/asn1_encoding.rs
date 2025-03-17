use crate::Result;
use super::*;

pub(crate) trait Asn1Encoding {
    fn encode(&self, writer: &mut Asn1Write) -> Result<usize>;
    fn get_length(&self) -> usize;
}
