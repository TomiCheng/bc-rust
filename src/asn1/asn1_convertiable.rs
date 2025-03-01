use super::Asn1Encodable;
use std::rc::Rc;

pub trait Asn1Convertiable {
    fn to_asn1_encodable(self: &Rc<Self>) -> Box<dyn Asn1Encodable>;
}
