use crate::asn1::Asn1String;
use crate::Result;

#[derive(Clone, Debug)]
pub struct Asn1Ia5String {
    contents: Vec<u8>,
}

impl Asn1Ia5String {
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        
        // TODO
        Ok(Asn1Ia5String { contents })
    }
}

impl Asn1String for Asn1Ia5String {
    fn to_asn1_string(&self) -> Result<String> {
        Ok(String::from_utf8(self.contents.clone())?)
    }
}