use std::ops::Index;
use crate::asn1::{Asn1EncodableVector, Asn1Object};
use crate::Result;

#[derive(Clone, Debug)]
pub struct Asn1Sequence {
    elements: Vec<Asn1Object>,
}


impl Asn1Sequence {
    pub fn new(elements: Vec<Asn1Object>) -> Self {
        Asn1Sequence { elements }
    }
    pub(crate) fn from_vector(vector: Asn1EncodableVector) -> Result<Asn1Sequence> {
        Ok(Asn1Sequence {
            elements: vector.get_elements().to_vec(),
        })
    }
    pub fn len(&self) -> usize {
        self.elements.len()
    }
}

impl Index<usize> for Asn1Sequence {
    type Output = Asn1Object;
    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}