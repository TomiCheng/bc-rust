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
    pub(crate) fn from_asn1_object(ans1_object: Asn1Object) -> Result<Self> {
        if let Ok(set) = ans1_object.try_into() {
            return Ok(set);
        }
        todo!()
    }
    pub fn get_elements(&self) -> &[Asn1Object] {
        &self.elements
    }
}

impl IntoIterator for Asn1Set {
    type Item = Asn1Object;
    type IntoIter = std::vec::IntoIter<Asn1Object>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}