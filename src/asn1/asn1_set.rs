use crate::Result;
use crate::asn1::EncodingType::*;
use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_write::get_contents_encodings;
use crate::asn1::constructed_dl_encoding::ConstructedDlEncoding;
use crate::asn1::constructed_il_encoding::ConstructedIlEncoding;
use crate::asn1::{Asn1EncodableVector, Asn1Object, EncodingType, asn1_tags, Asn1Encodable};
use std::hash::{Hash, Hasher};
use std::collections::HashMap;
#[derive(Clone, Debug, PartialEq)]
pub struct Asn1Set {
    elements: Vec<Asn1Object>,
}

impl Asn1Set {
    pub fn new(elements: Vec<Asn1Object>) -> Self {
        Asn1Set { elements }
    }
    pub(crate) fn from_vector(vector: Asn1EncodableVector) -> Result<Self> {
        Ok(Asn1Set {
            elements: vector.get_elements().to_vec(),
        })
    }
    pub(crate) fn from_asn1_object(ans1_object: Asn1Object) -> Result<Self> {
        if let Ok(set) = ans1_object.try_into() {
            return Ok(set);
        }
        Err(crate::BcError::with_invalid_format("Invalid Asn1Set format"))
    }
    pub fn get_elements(&self) -> &[Asn1Object] {
        &self.elements
    }
    fn create_sorted_der_encodings(&self) -> Vec<Box<dyn Asn1Encoding>> {
        let mut map = HashMap::new();
        for element in self.elements.iter() {
            let encoding = element.get_encoding(Der);
            let buffer  = element.get_der_encoded().unwrap_or_else(|_| vec![0u8; 0]);
            map.insert(buffer, encoding);
        }

        let mut pairs: Vec<_> = map.into_iter().collect();
        pairs.sort_by(|(a, _), (b, _)| a.cmp(b));
        let values = pairs.into_iter().map(|(_, encoding)| encoding).collect();
        values
    }
}
impl IntoIterator for Asn1Set {
    type Item = Asn1Object;
    type IntoIter = std::vec::IntoIter<Asn1Object>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}
impl Hash for Asn1Set {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for element in self.elements.iter() {
            element.hash(state)
        }
    }
}
impl Asn1EncodingInternal for Asn1Set {
    fn get_encoding(&self, encoding_type: EncodingType) -> Box<dyn Asn1Encoding> {
        match encoding_type {
            Dl => Box::new(ConstructedDlEncoding::new(
                asn1_tags::UNIVERSAL,
                asn1_tags::SET,
                get_contents_encodings(encoding_type, &self.elements),
            )),
            Ber => Box::new(ConstructedIlEncoding::new(
                asn1_tags::UNIVERSAL,
                asn1_tags::SET,
                get_contents_encodings(encoding_type, &self.elements),
            )),
            Der => Box::new(ConstructedDlEncoding::new(
                asn1_tags::UNIVERSAL,
                asn1_tags::SET,
                self.create_sorted_der_encodings(),
            )),
        }
    }
}
