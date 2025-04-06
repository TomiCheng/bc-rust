use crate::asn1::Asn1Object;

pub trait Asn1Convertible {
    fn to_asn1_object(&self) -> Asn1Object;
}
