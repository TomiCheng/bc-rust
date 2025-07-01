use crate::asn1::Asn1OctetString;

pub struct X509Extension {
    critical: bool, 
    value: Asn1OctetString,
}