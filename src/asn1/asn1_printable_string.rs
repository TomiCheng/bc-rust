use crate::asn1::Asn1String;
use crate::Result;

#[derive(Clone, Debug)]
pub struct Asn1PrintableString {
    contents: Vec<u8>,
}

impl Asn1PrintableString {
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        // TODO
        Ok(Asn1PrintableString { contents })
    }
}

impl Asn1String for Asn1PrintableString {
    fn to_asn1_string(&self) -> Result<String> {
        Ok(String::from_utf8(self.contents.clone())?)
    }
}