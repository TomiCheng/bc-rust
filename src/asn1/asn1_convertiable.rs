use std::sync;
use std::any;
use std::fmt;

use super::*;

pub trait Asn1Convertiable: fmt::Debug {
    fn as_any(&self) -> sync::Arc<dyn any::Any>;
    fn to_asn1_object(&self) -> sync::Arc<dyn Asn1Object>;
}
