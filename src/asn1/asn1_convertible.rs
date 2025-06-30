use crate::asn1::Asn1Object;
use crate::Result;

pub trait Asn1Convertible {
    fn to_asn1_object(&self) -> Result<Asn1Object>;
}