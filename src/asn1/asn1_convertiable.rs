use std::any::Any;

use super::Asn1Object;

pub trait Asn1Convertiable: Any {
    fn to_asn1_object(&self) -> Box<dyn Asn1Object>;
}

