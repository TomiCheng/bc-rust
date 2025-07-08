use crate::asn1::Asn1ObjectIdentifier;
pub struct KeyPurposeId {
    content: Asn1ObjectIdentifier
}

impl KeyPurposeId {
    pub fn new(content: Asn1ObjectIdentifier) -> Self {
        Self { content }
    }
}