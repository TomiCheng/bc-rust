use super::Asn1Object;

pub trait Asn1Convertiable {
    fn to_asn1_object(&self) -> Asn1Object;
}

