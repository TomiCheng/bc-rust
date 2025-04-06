use crate::Result;

#[derive(Clone, Debug)]
pub struct Asn1OctetString {
}
impl Asn1OctetString {
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        // TODO[asn1] check for zero length
        Ok(Asn1OctetString {})
    }
}