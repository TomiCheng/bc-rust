use crate::asn1::{Asn1Object, Asn1OctetString};
use crate::{BcError, Result};

pub struct SubjectKeyIdentifier {
    key_identifier: Vec<u8>,
}

impl SubjectKeyIdentifier {
    pub fn new(key_identifier: Vec<u8>) -> Self {
        SubjectKeyIdentifier { key_identifier }
    }
    pub fn with_octet_string(octet_string: Asn1OctetString) -> Result<Self> {
        Ok(Self::new(octet_string.into_vec()))
    }
    pub fn key_identifier(&self) -> &[u8] {
        &self.key_identifier
    }
}
impl TryFrom<Asn1Object> for SubjectKeyIdentifier {
    type Error = BcError;

    fn try_from(value: Asn1Object) -> Result<Self> {
        if let Asn1Object::OctetString(v) = value {
            return Self::with_octet_string(v);
        }
        todo!();
    }
}
