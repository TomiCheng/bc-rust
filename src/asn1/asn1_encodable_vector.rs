use crate::asn1::Asn1Object;
use std::ops::Index;

pub struct Asn1EncodableVector {
    elements: Vec<Asn1Object>,
}

impl Asn1EncodableVector {}

impl Asn1EncodableVector {
    const DEFAULT_CAPACITY: usize = 10;
    pub fn new(elements: Vec<Asn1Object>) -> Self {
        Asn1EncodableVector { elements }
    }
    pub fn empty() -> Self {
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
    pub(crate) fn len(&self) -> usize {
        self.elements.len()
    }
}

impl Index<usize> for Asn1EncodableVector {
    type Output = Asn1Object;

    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}
