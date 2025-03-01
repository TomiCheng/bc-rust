use super::asn1_encoding::Asn1Encoding;
use super::asn1_object::Asn1ObjectInternal;
use super::asn1_tags::{NULL, UNIVERSAL};
use super::asn1_write::EncodingType;
use super::primitive_encoding::PrimitiveEncoding;
use super::{Asn1Convertiable, Asn1Encodable, Asn1ObjectImpl};
use std::fmt::{Display, Formatter};
use std::rc::Rc;

/// A Null object.
pub struct DerNull;

impl DerNull {
    pub fn new() -> Self {
        DerNull {}
    }
}

impl Asn1ObjectInternal for DerNull {
    fn get_encoding_with_type(&self, _encoding: &EncodingType) -> Box<dyn Asn1Encoding> {
        Box::new(PrimitiveEncoding::new(UNIVERSAL, NULL, Rc::new(vec![])))
    }
}

impl Asn1Convertiable for DerNull {
    fn to_asn1_encodable(self: &Rc<Self>) -> Box<dyn Asn1Encodable> {
        Box::new(Asn1ObjectImpl::new(self.clone()))
    }
}

impl Display for DerNull {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "NULL")
    }
}
