use crate::Result;
use crate::asn1::EncodingType::*;
use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_write::get_contents_encodings;
use crate::asn1::constructed_dl_encoding::ConstructedDlEncoding;
use crate::asn1::constructed_il_encoding::ConstructedIlEncoding;
use crate::asn1::{Asn1Encodable, Asn1EncodableVector, Asn1Object, EncodingType, asn1_tags};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
#[derive(Clone, Debug, PartialEq)]
pub struct Asn1Set {
    elements: Vec<Asn1Object>,
}

impl Asn1Set {
    pub fn new(elements: Vec<Asn1Object>) -> Self {
        Asn1Set { elements }
    }
    pub(crate) fn from_vector(vector: Asn1EncodableVector) -> Result<Self> {
        Ok(Asn1Set::new(vector.get_elements().to_vec()))
    }
    pub fn from_vector_sorted(vector: Asn1EncodableVector, sort: bool) -> Result<Self> {
        let mut elements = vector.get_elements().to_vec();
        if sort {
            let mut map = HashMap::new();
            for element in elements.into_iter() {
                let buffer = element.get_der_encoded().unwrap_or_else(|_| vec![0u8; 0]);
                map.insert(buffer, element);
            }
            let mut pairs: Vec<_> = map.into_iter().collect();
            pairs.sort_by(|(a, _), (b, _)| a.cmp(b));
            elements = pairs.into_iter().map(|(_, element)| element).collect();
        }
        Ok(Asn1Set::new(elements))
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
            let buffer = element.get_der_encoded().unwrap_or_else(|_| vec![0u8; 0]);
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

    fn get_encoding_implicit(&self, encoding_type: EncodingType, tag_class: u8, tag_no: u8) -> Box<dyn Asn1Encoding> {
        match encoding_type {
            Dl => Box::new(ConstructedDlEncoding::new(
                tag_class,
                tag_no,
                get_contents_encodings(encoding_type, &self.elements),
            )),
            Ber => Box::new(ConstructedIlEncoding::new(
                tag_class,
                tag_no,
                get_contents_encodings(encoding_type, &self.elements),
            )),
            Der => Box::new(ConstructedDlEncoding::new(tag_class, tag_no, self.create_sorted_der_encodings())),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::asn1::EncodingType::Der;
    use crate::asn1::try_from_tagged::TryIntoTagged;
    use crate::asn1::{
        Asn1BitString, Asn1Boolean, Asn1Encodable, Asn1EncodableVector, Asn1Integer, Asn1Object, Asn1OctetString, Asn1Sequence, Asn1Set,
        Asn1TaggedObject,
    };

    #[test]
    fn test_01() {
        let mut v = Asn1EncodableVector::empty();
        v.push(Asn1OctetString::new(vec![0u8; 10]).into());
        v.push(Asn1BitString::with_pad_bits(&vec![0u8; 10], 0).unwrap().into());
        v.push(Asn1Integer::with_i64(100).into());
        v.push(Asn1Boolean::new(true).into());
        check_sorted_set(Asn1Set::from_vector_sorted(v, true).unwrap());
    }
    #[test]
    fn test_02() {
        let mut v = Asn1EncodableVector::empty();
        v.push(Asn1Integer::with_i64(100).into());
        v.push(Asn1Boolean::new(true).into());
        v.push(Asn1BitString::with_pad_bits(&vec![0u8; 10], 0).unwrap().into());
        v.push(Asn1OctetString::new(vec![0u8; 10]).into());
        check_sorted_set(Asn1Set::from_vector_sorted(v, true).unwrap());
    }
    #[test]
    fn test_03() {
        let mut v = Asn1EncodableVector::empty();
        v.push(Asn1Boolean::new(true).into());
        v.push(Asn1OctetString::new(vec![0u8; 10]).into());
        v.push(Asn1BitString::with_pad_bits(&vec![0u8; 10], 0).unwrap().into());
        v.push(Asn1Integer::with_i64(100).into());
        check_sorted_set(Asn1Set::from_vector_sorted(v, true).unwrap());
    }
    #[test]
    fn test_04() {
        let mut v = Asn1EncodableVector::empty();
        v.push(Asn1BitString::with_pad_bits(&vec![0u8; 10], 0).unwrap().into());
        v.push(Asn1OctetString::new(vec![0u8; 10]).into());
        v.push(Asn1Integer::with_i64(100).into());
        v.push(Asn1Boolean::new(true).into());
        check_sorted_set(Asn1Set::from_vector_sorted(v, true).unwrap());
    }
    #[test]
    fn test_05() {
        let mut v = Asn1EncodableVector::empty();
        v.push(Asn1OctetString::new(vec![0u8; 10]).into());
        v.push(Asn1BitString::with_pad_bits(&vec![0u8; 10], 0).unwrap().into());
        v.push(Asn1Integer::with_i64(100).into());
        v.push(Asn1Boolean::new(true).into());
        let set = Asn1Set::from_vector_sorted(v, false).unwrap();
        assert!(set.get_elements()[0].is_octet_string());
    }
    #[test]
    fn test_06() {
        let mut v = Asn1EncodableVector::empty();
        v.push(Asn1OctetString::new(vec![0u8; 10]).into());
        v.push(Asn1BitString::with_pad_bits(&vec![0u8; 10], 0).unwrap().into());
        v.push(Asn1Integer::with_i64(100).into());
        v.push(Asn1Boolean::new(true).into());

        let tag = Asn1TaggedObject::from_explicit_tag_object(false, 1, Asn1Sequence::from_vector(v).unwrap().into());
        let buffer = tag.get_encoded(Der).unwrap();
        let tag: Asn1TaggedObject = Asn1Object::with_bytes(&buffer).unwrap().try_into().unwrap();
        let set: Asn1Sequence = tag.try_into_tagged(false).unwrap();
        assert!(!set.get_elements()[0].is_boolean());
    }
    #[test]
    fn test_equality_01() {
        let mut v = Asn1EncodableVector::empty();
        v.push(Asn1Boolean::new(true).into());
        v.push(Asn1Boolean::new(true).into());
        v.push(Asn1Boolean::new(true).into());

        Asn1Set::from_vector(v).unwrap();
    }
    fn check_sorted_set(value: Asn1Set) {
        let elements = value.get_elements();
        assert_eq!(elements.len(), 4);
        assert!(elements[0].is_boolean());
        assert!(elements[1].is_integer());
        assert!(elements[2].is_bit_string());
        assert!(elements[3].is_octet_string());
    }
}
