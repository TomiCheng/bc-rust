use crate::asn1::{Asn1Object, Asn1TaggedObject};
use crate::asn1::x509::X509Extensions;
use crate::Result;

pub struct Asn1EncodableVector {
    elements: Vec<Asn1Object>,
}

impl Asn1EncodableVector {

}

impl Asn1EncodableVector {
    const DEFAULT_CAPACITY: usize = 10;
    pub fn new() -> Self {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }
    pub(crate) fn with_capacity(initial_capacity: usize) -> Self {
        Asn1EncodableVector {
            elements: Vec::with_capacity(initial_capacity),
        }
    }
    pub(crate) fn add(&mut self, element: Asn1Object) {
        self.elements.push(element);
    }
    pub(crate) fn get_elements(&self) -> &[Asn1Object] {
        &self.elements
    }
    pub(crate) fn optional_tagged(&mut self, is_explicit: bool, tag_no: u8, asn1_object: &Asn1Object) -> Result<()> {
        //self.elements.push(Asn1Object::from(Asn1TaggedObject::with_context_specific(is_explicit, tag_no, asn1_object.clone())))
        todo!()
    }
}