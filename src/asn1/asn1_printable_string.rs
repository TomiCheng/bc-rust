use crate::Result;

#[derive(Clone, Debug)]
pub struct Asn1PrintableString {}

impl Asn1PrintableString {
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        // TODO
        Ok(Asn1PrintableString {})
    }
}