use crate::asn1::Asn1OctetString;

pub struct X509Extension {
    critical: bool, 
    value: Asn1OctetString,
}

impl X509Extension {
    pub fn new(critical: bool, value: Asn1OctetString) -> Self {
        X509Extension { critical, value }
    }
    pub fn is_critical(&self) -> bool {
        self.critical
    }
    pub fn get_value(&self) -> &Asn1OctetString {
        &self.value
    }
}