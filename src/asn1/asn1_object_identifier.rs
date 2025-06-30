use crate::asn1::Asn1Object;
use crate::Result;

#[derive(Clone, Debug)]
pub struct Asn1ObjectIdentifier {
    contents: Vec<u8>,
    
}

impl Asn1ObjectIdentifier {
    pub fn new(contents: Vec<u8>) -> Self {
        Asn1ObjectIdentifier { contents }
    }
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        // todo!()
        Ok(Asn1ObjectIdentifier { contents })
    }
    pub(crate) fn from_asn1_object(asn1_object: &Asn1Object) -> Result<Self> {
        if let Some(object_identifier) = asn1_object.as_object_identifier() {
            Ok(object_identifier.clone())
        } else {
            Err(crate::BcError::with_invalid_cast("Expected an octet string for Asn1ObjectIdentifier"))
        }
    }
}