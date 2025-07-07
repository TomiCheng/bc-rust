use std::ops::Index;
use crate::asn1::{Asn1EncodableVector, Asn1Object, Asn1TaggedObject};
use crate::asn1::asn1_universal_type::Asn1UniversalType;
use crate::asn1::try_from_tagged::TryFromTagged;
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
    pub(crate) fn from_asn1_object(asn1_object: Asn1Object) -> Result<Self> {
        if let Some(sequence) = asn1_object.as_sequence() {
            return Ok(sequence.clone());
        }
        todo!()
    }
    pub fn len(&self) -> usize {
        self.elements.len()
    }
    pub fn get_elements(&self) -> &[Asn1Object] {
        &self.elements
    }
    pub fn get_tagged(tagged_object: Asn1TaggedObject, declared_explicit: bool) -> Result<Self> {
        let metadata = Asn1SequenceMetadata::new();
        metadata.get_tagged(tagged_object, declared_explicit)
    }
}
impl Index<usize> for Asn1Sequence {
    type Output = Asn1Object;
    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}

impl From<Asn1Sequence> for Vec<Asn1Object> {
    fn from(value: Asn1Sequence) -> Self {
        value.elements
    }
}

impl IntoIterator for Asn1Sequence {
    type Item = Asn1Object;
    type IntoIter = std::vec::IntoIter<Asn1Object>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

pub(crate) struct Asn1SequenceMetadata;

impl Asn1SequenceMetadata {
    pub(crate) fn new() -> Self {
        Asn1SequenceMetadata
    }
}
impl Asn1UniversalType<Asn1Sequence> for Asn1SequenceMetadata {
    fn checked_cast(&self, asn1_object: Asn1Object) -> Result<Asn1Sequence> {
        asn1_object.try_into()
        // if let Ok(value) =  {
        //     Ok(value)
        // } else {
        //     Err(crate::BcError::with_invalid_operation("Expected an ASN.1 Integer object"))
        // }
    }

    fn implicit_constructed(&self, sequence: Asn1Sequence) -> Result<Asn1Sequence> {
        todo!();
    }
}
impl TryFromTagged for Asn1Sequence {
    fn try_from_tagged(tagged: Asn1TaggedObject, declared_explicit: bool) -> Result<Self>
    where
        Self: Sized,
    {
        let metadata = Asn1SequenceMetadata::new();
        metadata.get_tagged(tagged, declared_explicit)
    }
}
