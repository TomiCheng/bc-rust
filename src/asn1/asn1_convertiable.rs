use std::sync;

use super::*;

pub trait Asn1Convertiable {
    fn to_asn1_object(&self) -> sync::Arc<dyn Asn1Object>;
}
