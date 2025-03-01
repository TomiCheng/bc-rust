//use super::asn1_convertiable::Asn1ConvertiableInternal;
use crate::Result;
use std::io::Write;

pub const BER: &str = "BER";
pub const DER: &str = "DER";
pub const DL: &str = "DL";

pub trait Asn1Encodable {
    fn encode_to(&self, writer: &mut dyn Write) -> Result<usize>;
    fn encode_to_with_encoding(&self, writer: &mut dyn Write, encoding: &str) -> Result<usize>;
    fn get_encoded(&self) -> Result<Vec<u8>>;
    fn get_encoded_with_encoding(&self, encoding: &str) -> Result<Vec<u8>>;
    fn get_der_encoded(&self) -> Option<Vec<u8>> {
        self.get_encoded_with_encoding(DER).ok()
    }
}

//pub(crate) trait Asn1EncodableInternal: Asn1ConvertiableInternal {
    //fn as_asn1_convertiable_internal(&self) -> &dyn Asn1ConvertiableInternal;
    //fn get_encoded_alloc(&self, encoding: &str, pre_alloc: usize, post_alloc: usize) -> Result<Vec<u8>>
    //fn get_encoded(&self) -> Result<Vec<u8>>;
//}

// pub struct Asn1EncodableImpl<'a> {
//     instance: &'a dyn Asn1EncodableInternal,
// }

// impl<'a> Asn1EncodableImpl<'a> {
//     pub(crate) fn new(instance: &'a dyn Asn1EncodableInternal) -> Self {
//         Asn1EncodableImpl { instance }
//     }
//     // pub fn get_encoded_alloc(&self, encoding: &str, pre_alloc: usize, post_alloc: usize) -> Result<Vec<u8>> {
//     //     //self.to_asn1_object_internal().get_encoded_alloc(encoding, pre_alloc, post_alloc)
//     //     self.instance.to_asn1_object_internal().get_encoded_alloc(encoding, pre_alloc, post_alloc)
//     //     //self.instance
//     //     //Asn1ObjectImpl::new(self.instance.as_asn1_object_internal()).get_encoded_alloc(encoding, pre_alloc, post_alloc)
//     //     //Ok(vec![])
//     // }
// }

// impl<'a> Asn1Encodable for Asn1EncodableImpl<'a>  {
//     // fn get_encoded(&self) -> Result<Vec<u8>> {
//     //     self.get_encoded_alloc(BER, 0, 0)
//     // }
// }

// impl<'a> Asn1EncodableInternal for Asn1EncodableImpl<'a>  {
//     fn as_asn1_convertiable_internal(&self) -> &dyn Asn1ConvertiableInternal {
//         todo!()
//     }

//     // fn get_encoded_alloc(&self, encoding: &str, pre_alloc: usize, post_alloc: usize) -> Result<Vec<u8>> {
//     //     todo!()
//     // }
// }

// pub trait Asn1Encodable: Asn1Convertiable {
//     fn encode_to(&self, writer: &mut dyn Write) -> Result<usize> {
//          self.to_asn1_object().encode_to(writer)
//     }
//     fn encode_to_with_encoding(&self, writer: &mut dyn Write, encoding: &str) -> Result<usize> {
//         self.to_asn1_object().encode_to_with_encoding(writer, encoding)
//     }
//     fn get_encoded(&self) -> Result<Vec<u8>>;
//     fn get_encoded_with_encoding(&self, encoding: &str) -> Result<Vec<u8>>;
//     fn get_der_encoded(&self) -> Option<Vec<u8>> {
//         self.get_encoded_with_encoding(DER).ok()
//     }
// }

// pub(crate) trait Asn1EncodableInternal {
//     fn get_encoded_alloc(&self, encoding: &str, pre_alloc: usize, post_alloc: usize) -> Result<Vec<u8>>;
// }

// pub fn get_encoded<T: Asn1EncodableInternal>(instance : &T) -> Result<Vec<u8>> {
//     instance.get_encoded_alloc(BER, 0, 0)
// }

// pub fn get_encoded_with_encoding<T: Asn1EncodableInternal>(instance : &T, encoding: &str) -> Result<Vec<u8>> {
//     instance.get_encoded_alloc(encoding, 0, 0)
// }

// pub(crate) trait Asn1EncodableInternal {
//     fn encode_to(&self, writer: &mut dyn Write) -> Result<usize>;
// }

// pub struct Asn1EncodableImpl<T: Asn1ConvertiableInternal> {
//     instance: Box<T>,
// }

// impl<T: Asn1ConvertiableInternal> Asn1EncodableImpl<T> {
//     pub fn new(instance: Box<T>) -> Self {
//         Asn1EncodableImpl { instance }
//     }
//     fn encode_to(&self, writer: &mut dyn Write) -> Result<usize> {
//         self.instance.to_asn1_object_internal().encode_to(writer)
//     }
// }
// impl<'a> Asn1EncodableInternal for Asn1EncodableImpl<'a> {
//     fn encode_to(&self, writer: &mut dyn Write) -> Result<usize> {
//         let asn1_object_internal = self.parent.to_asn1_object_internal();
//         asn1_object_internal.encode_to(writer)
//     }
// }

// pub struct Asn1ObjectImpl<'a> {
//     parent: &'a Asn1EncodableImpl<'a>,
// }

// impl<'a> Asn1ObjectImpl<'a> {
//     fn new(parent: &'a Asn1EncodableImpl<'a>) -> Self {
//         Asn1ObjectImpl { parent }
//     }
//     fn encode_to(&self, writer: &mut dyn Write) -> Result<usize> {
//         let asn1_object = self.parent.to_asn1_object_internal();
//         asn1_object.encode_to(writer)
//     }
// }
