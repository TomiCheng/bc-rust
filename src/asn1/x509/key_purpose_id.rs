use crate::asn1::Asn1ObjectIdentifier;
use crate::Result;
pub struct KeyPurposeId {
    content: Asn1ObjectIdentifier
}

impl KeyPurposeId {
    pub(crate) fn new(content: Asn1ObjectIdentifier) -> Self {
        Self { content }
    }
}