use crate::asn1::{Asn1EncodableVector, Asn1Object};
use crate::Result;
#[derive(Clone, Debug)]
pub struct Asn1Set {
    elements: Vec<Asn1Object>
}

impl Asn1Set {
    pub(crate) fn from_vector(vector: Asn1EncodableVector) -> Result<Self> {
        Ok(Asn1Set {
            elements: vector.get_elements().to_vec(),
        })
    }
    pub(crate) fn from_asn1_object(ans1_object: &Asn1Object) -> Result<Self> {
        if let Some(set) = ans1_object.as_set() {
            return Ok(set.clone());
        }
        todo!()
    }
    pub fn get_elements(&self) -> &[Asn1Object] {
        &self.elements
    }
}