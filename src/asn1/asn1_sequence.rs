use std::hash::{Hash, Hasher};
use std::ops::Index;
use crate::asn1::{asn1_tags, Asn1EncodableVector, Asn1Object, Asn1TaggedObject, EncodingType};
use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_universal_type::Asn1UniversalType;
use crate::asn1::asn1_write::get_contents_encodings;
use crate::asn1::constructed_dl_encoding::ConstructedDlEncoding;
use crate::asn1::EncodingType::Der;
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
    pub fn len(&self) -> usize {
        self.elements.len()
    }
    pub fn get_elements(&self) -> &[Asn1Object] {
        &self.elements
    }
    // pub fn get_tagged(tagged_object: Asn1TaggedObject, declared_explicit: bool) -> Result<Self> {
    //     let metadata = Asn1SequenceMetadata::new();
    //     metadata.get_tagged(tagged_object, declared_explicit)
    // }
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
impl Hash for Asn1Sequence {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for element in self.elements.iter() {
            element.hash(state)
        }
    }
}
impl TryFromTagged for Asn1Sequence {
    fn try_from_tagged(tagged: Asn1TaggedObject, declared_explicit: bool) -> Result<Self>
    where
        Self: Sized,
    {
        tagged.try_from_base_universal(declared_explicit, Asn1SequenceMetadata)
    }
}
impl Asn1EncodingInternal for Asn1Sequence {
    fn get_encoding(&self, _: EncodingType) -> Box<dyn Asn1Encoding> {
        Box::new(ConstructedDlEncoding::new(asn1_tags::UNIVERSAL, asn1_tags::SEQUENCE, get_contents_encodings(Der, &self.elements)))
    }
}

struct Asn1SequenceMetadata;

impl Asn1UniversalType<Asn1Sequence> for Asn1SequenceMetadata {
    fn checked_cast(&self, asn1_object: Asn1Object) -> Result<Asn1Sequence> {
        asn1_object.try_into()
    }
}