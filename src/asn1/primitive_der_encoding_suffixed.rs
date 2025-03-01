use super::asn1_write::get_length_of_encoding_dl;
use super::asn1_write_impl::Asn1WriteImpl;
use super::der_encoding::DerEncodingImpl;
use super::primitive_der_encoding::PrimitiveDerEncoding;
use super::Asn1Write;
use std::any::Any;
use std::cmp::{Ordering, PartialEq, PartialOrd};

pub(crate) struct PrimitiveDerEncodingSuffixed {
    parent: DerEncodingImpl,
    contents_octets: Vec<u8>,
    contents_suffix: u8,
}

impl PrimitiveDerEncodingSuffixed {
    pub(crate) fn new(
        tag_class: u32,
        tag_no: u32,
        contents_octets: &[u8],
        contents_suffix: u8,
    ) -> Self {
        assert!(contents_octets.len() > 0);

        PrimitiveDerEncodingSuffixed {
            parent: DerEncodingImpl::new(tag_class, tag_no),
            contents_octets: contents_octets.to_vec(),
            contents_suffix,
        }
    }
    pub fn get_length(&self) -> usize {
        get_length_of_encoding_dl(self.parent.get_tag_no(), self.contents_octets.len())
    }
    pub fn encode<T: Any + Asn1Write>(&self, writer: &mut T) {
        let writer_any = writer as &mut dyn Any;
        if let Some(writer) = writer_any.downcast_mut::<Asn1WriteImpl>() {
            writer.write_identifier(self.parent.get_tag_class(), self.parent.get_tag_no());
            writer.write_dl(self.contents_octets.len() as u32);
            writer.write(&self.contents_octets[0..(self.contents_octets.len() - 1)]);
            writer.write_u8(self.contents_suffix);
            return;
        }
    }
    pub(crate) fn compare_length_and_contents<T: Any>(&self, other: &T) -> Option<Ordering> {
        let other_any = other as &dyn Any;
        if let Some(other) = other_any.downcast_ref::<PrimitiveDerEncodingSuffixed>() {
            return Some(compare_suffixed(
                &self.contents_octets,
                self.contents_suffix,
                &other.contents_octets,
                other.contents_suffix,
            ));
        } else if let Some(other) = other_any.downcast_ref::<PrimitiveDerEncoding>() {
            let other_contents_octets = other.get_contetns_octets();
            let length = other_contents_octets.len();
            if length == 0 {
                return Some(Ordering::Greater);
            }
            let other_contents_suffix = other_contents_octets[length - 1];
            return Some(compare_suffixed(
                &self.contents_octets,
                self.contents_suffix,
                other_contents_octets,
                other_contents_suffix,
            ));
        } else {
            return None;
        }
    }
}

impl PartialOrd for PrimitiveDerEncodingSuffixed {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let mut result = self.parent.partial_cmp(&other.parent);
        if result.is_none() {
            result = self.compare_length_and_contents(other);
        }
        return result;
    }
}

impl PartialEq for PrimitiveDerEncodingSuffixed {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

fn compare_suffixed(octets_a: &[u8], suffix_a: u8, octets_b: &[u8], suffix_b: u8) -> Ordering {
    assert!(octets_a.len() > 0);
    assert!(octets_b.len() > 0);

    let length = octets_a.len();
    if length != octets_b.len() {
        return length.cmp(&octets_b.len());
    }

    let last = length - 1;
    let ordering = octets_a[0..last].cmp(&octets_b[0..last]);
    if ordering != Ordering::Equal {
        return ordering;
    }
    return suffix_a.cmp(&suffix_b);
}
