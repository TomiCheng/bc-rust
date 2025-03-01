use super::asn1_encoding::Asn1Encoding;
use super::asn1_tags::PRIVATE;
use std::cmp::{Ordering, PartialEq, PartialOrd};

pub(crate) trait DerEncoding: Asn1Encoding {}

pub(crate) struct DerEncodingImpl {
    tag_class: u32,
    tag_no: u32,
}

impl DerEncodingImpl {
    pub(crate) fn new(tag_class: u32, tag_no: u32) -> Self {
        assert_eq!(tag_class & PRIVATE, tag_class);
        DerEncodingImpl { tag_class, tag_no }
    }
    pub(crate) fn get_tag_class(&self) -> u32 {
        self.tag_class
    }
    pub(crate) fn get_tag_no(&self) -> u32 {
        self.tag_no
    }
    pub(crate) fn compare_to<F>(
        &self,
        other: &Self,
        compare_length_and_contents: F,
    ) -> Option<Ordering>
    where
        F: FnOnce() -> Option<Ordering>,
    {
        if self.tag_class != other.tag_class {
            return Some(self.tag_class.cmp(&other.tag_class));
        } else if self.tag_no != other.tag_no {
            return Some(self.tag_no.cmp(&other.tag_no));
        } else {
            return compare_length_and_contents();
        }
    }
}

impl PartialOrd for DerEncodingImpl {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.tag_class != other.tag_class {
            return Some(self.tag_class.cmp(&other.tag_class));
        } else if self.tag_no != other.tag_no {
            return Some(self.tag_no.cmp(&other.tag_no));
        } else {
            return None;
        }
    }
}

impl PartialEq for DerEncodingImpl {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}
