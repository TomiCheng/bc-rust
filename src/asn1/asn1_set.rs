use crate::asn1::{Asn1EncodableVector, Asn1Object, Asn1Sequence};
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
}