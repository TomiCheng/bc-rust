use super::asn1_encoding::Asn1Encoding;
//use super::asn1_write::get_length_of_encoding_dl;
//use super::asn1_write_impl::Asn1WriteImpl;
use super::der_encoding::{DerEncoding, DerEncodingImpl};
//use super::primitive_der_encoding_suffixed::PrimitiveDerEncodingSuffixed;
use super::Asn1Write;
use crate::{BcError, Result};
use std::any::Any;
use std::cmp::Ordering;

pub(crate) struct PrimitiveDerEncoding {
    parent: DerEncodingImpl,
    contents_octets: Vec<u8>,
}

impl PrimitiveDerEncoding {
    pub(crate) fn new(tag_class: u32, tag_no: u32, contents_octets: &[u8]) -> Self {
        PrimitiveDerEncoding {
            parent: DerEncodingImpl::new(tag_class, tag_no),
            contents_octets: contents_octets.to_vec(),
        }
    }
    //     pub(crate) fn get_contetns_octets(&self) -> &[u8] {
    //         &self.contents_octets
    //     }

     //    pub(crate) fn compare_length_and_contents<T: Any>(&self, other: &T) -> Option<Ordering> {
    //         let other_any = other as &dyn Any;
    //         if let Some(other) = other_any.downcast_ref::<PrimitiveDerEncodingSuffixed>() {
    //             return other
    //                 .compare_length_and_contents(self)
    //                 .map(|ordering| ordering.reverse());
    //         } else if let Some(other) = other_any.downcast_ref::<PrimitiveDerEncoding>() {
    //             let length = self.contents_octets.len();
    //             if length != other.contents_octets.len() {
    //                 return length.partial_cmp(&other.contents_octets.len());
    //             }
    //             return self.contents_octets.partial_cmp(&other.contents_octets);
    //         } else {
     //            return None;
    //         }
    //     }
}

// impl PartialOrd<dyn DerEncoding> for PrimitiveDerEncoding {
//     fn partial_cmp(&self, other: &dyn DerEncoding) -> Option<Ordering> {
//         let mut result = self.parent.partial_cmp(&other.parent);
//         if result.is_none() {
//             result = self.compare_length_and_contents(other);
//         }
//         return result;
//     }
// }

// impl PartialEq<dyn DerEncoding> for PrimitiveDerEncoding {
//     fn eq(&self, other: &dyn DerEncoding) -> bool {
//         self.partial_cmp(other) == Some(Ordering::Equal)
//     }
// }

impl DerEncoding for PrimitiveDerEncoding {}

impl Asn1Encoding for PrimitiveDerEncoding {
    fn encode(&self, writer: &mut Asn1Write) -> Result<usize> {
    //     let mut length = 0;
    //     length += writer.write_identifier(self.parent.get_tag_class(), self.parent.get_tag_no())?;
    //     length += writer.write_dl(self.contents_octets.len() as u32)?;
    //     length += writer.write(&self.contents_octets)?;
    //     return Ok(length);
     todo!();
    }

    // fn get_length(&self) -> usize {
    //     get_length_of_encoding_dl(self.parent.get_tag_no(), self.contents_octets.len())
    // }
}
