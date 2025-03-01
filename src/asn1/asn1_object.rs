// use super::asn1_convertiable::Asn1ConvertiableInternal;
// // // use super::asn1_write_impl::Asn1WriteImpl;
// use super::asn1_write::EncodingType;
// use super::{asn1_encodable::Asn1EncodableImpl, asn1_encoding::Asn1Encoding};
// // use super::Asn1Encodable;
// // use super::{asn1_encodable::Asn1EncodableImpl, asn1_encoding::Asn1Encoding, der_encoding::DerEncoding};

use super::asn1_encodable::BER;
use super::asn1_encoding::Asn1Encoding;
use super::asn1_write::EncodingType;
use super::{Asn1Encodable, Asn1Write};
use crate::asn1::asn1_write::get_encoding_type;
use crate::Result;
use std::io::Write;
use std::rc::Rc;

// use super::asn1_encoding::Asn1Encoding;
// use super::asn1_write::EncodingType;
// use super::der_encoding::DerEncoding;
// use super::asn1_encodable::BER;
// // use super::asn1_encoding::Asn1Encoding;
// // use super::asn1_write::{get_encoding_type, EncodingType};
// // use super::der_encoding::DerEncoding;
// use super::{Asn1Encodable, Asn1Write};
// pub trait Asn1Object: Asn1Encodable {
//     fn encode_to(&self, writer: &mut dyn Write) -> Result<usize> {
//         todo!();
//     }
//     //fn as_any(&self) -> &dyn Any;
//     // fn encode_to(&self, writer: &mut dyn Write) -> Result<usize> {
//     //     let any = self.to_any();
//     //     let any1 = any.downcast_ref::<&dyn Asn1ObjectInternal>().unwrap();
//     //     return Ok(0);
//     // }
//     // fn encode_to_with_encoding(&self, writer: &mut dyn Write, encoding: &str) -> Result<usize> {
//     //     todo!();
//     // }
//     // fn encode_to(self: &Box<Self>, writer: &mut dyn Write) -> Result<usize> {
//     //     //let ddd = self as &dyn Asn1ObjectInternal;
//     //     todo!();
//     // }
//     //fn get_encoding_with_i32(&self, encoding: i32) -> Box<dyn Asn1Encoding>;
//     //fn encode_to(&self, writer: &mut dyn Write) -> Result<usize>;
//     //fn encode_to_with_encoding(&self, writer: Box<dyn Write>, encoding: &str);
//     fn encode_to(&self, writer: &mut dyn Write) -> Result<usize>;
// }

// pub(crate) trait Asn1ObjectInternal: Any {
//     fn as_any(&self) -> &dyn Any;
//     fn get_encoding_with_type(&self, encoding: &EncodingType) -> Box<dyn Asn1Encoding>;
//     fn get_encoding_with_implicit(
//         &self,
//         encoding: &EncodingType,
//         tag_class: u32,
//         tag_no: u32,
//     ) -> Box<dyn Asn1Encoding>;
//     fn get_encoding_der(&self) -> Box<dyn DerEncoding>;
//     fn get_encoding_der_implicit(&self, tag_class: u32, tag_no: u32) -> Box<dyn DerEncoding>;
//     fn asn1_equals(&self, asn1_object: &dyn Asn1ObjectInternal) -> bool;
//     fn asn1_get_hash_code(&self) -> u64;
//     // fn get_encoding_with_type(&self, encoding: &EncodingType) -> Box<dyn Asn1Encoding>;
//     // fn get_encoding_with_implicit(&self, encoding: &EncodingType, tag_class: u32, tag_no: u32) -> Box<dyn Asn1Encoding>;
//     // fn get_encoding_der(&self) -> Box<dyn DerEncoding>;
//     // fn get_encoding_der_implicit(&self, tag_class: u32, tag_no: u32) -> Box<dyn DerEncoding>;
//     // fn get_encoded_alloc(&self, encoding_code: &str, pre_alloc: usize, post_alloc: usize) -> Result<Vec<u8>> {
//     //     let encoding_type = get_encoding_type(encoding_code);
//     //     let encoding = self.get_encoding_with_type(&encoding_type);
//     //     let length = encoding.get_length();
//     //     let mut result = vec![0u8; pre_alloc + length + post_alloc];
//     //     result.resize(pre_alloc, 0);
//     //     let mut asn1_writer = Asn1Write::create_with_encoding(&mut result, encoding_code);
//     //     let writted_length = encoding.encode(&mut asn1_writer)?;
//     //     debug_assert_eq!(writted_length, length);
//     //     Ok(result)
//     // }
//     fn encode_to(&self, writer: &mut dyn Write) -> Result<usize> {
//         let mut asn1_writer = Asn1Write::create_with_encoding(writer, BER);
//         let encoding = self.get_encoding_with_type(asn1_writer.get_encoding());
//         Ok(encoding.encode(&mut asn1_writer)?)
//     }
// }

// // pub struct Asn1ObjectImpl<T> where T: Asn1Object {
// //     instance: T,
// // }

// // impl<T> Asn1ObjectImpl<T> where T: Asn1Object {
// //     pub fn encode_to(&self, writer: Box<dyn Write>) {
// //         let mut asn1_writer = Asn1Write::create(writer);
// //         self.instance.get_encoding_with_i32(asn1_writer.get_encoding()).encode(asn1_writer.as_mut());
// //     }
// //     pub fn encode_to_with_encoding(&self, writer: Box<dyn Write>, encoding: &str) {
// //         let mut asn1_writer = Asn1Write::create_with_encoding(writer, encoding);
// //         self.instance.get_encoding_with_i32(asn1_writer.get_encoding()).encode(asn1_writer.as_mut());
// //     }
// // }

// use super::asn1_encodable::BER;
// use super::asn1_encoding::Asn1Encoding;
// use super::asn1_write::EncodingType;
// use super::{Asn1Encodable, Asn1Write};

//pub trait Asn1Object: Asn1Encodable {}

//pub(crate) trait Asn1ObjectInternal {
//fn get_encoding_with_type(&self, encoding: &EncodingType) -> Box<dyn Asn1Encoding>;
//fn encode_to(&self, writer: &mut dyn Write) -> Result<usize>;
//}

// pub(crate) fn encode_to<T: Asn1ObjectInternal>(instance: &T ,writer: &mut dyn Write) -> Result<usize> {
//     let mut asn1_writer = Asn1Write::create_with_encoding(writer, BER);
//     instance.get_encoding_with_type(asn1_writer.get_encoding()).encode(&mut asn1_writer)
// }

// pub(crate) fn encode_to_with_encoding<T: Asn1ObjectInternal>(instance: &T ,writer: &mut dyn Write, encoding: &str) -> Result<usize> {
//     let mut asn1_writer = Asn1Write::create_with_encoding(writer, encoding);
//     instance.get_encoding_with_type(asn1_writer.get_encoding()).encode(&mut asn1_writer)
// }

// pub(crate) fn get_encoded_alloc<T: Asn1ObjectInternal>(instance: &T, encoding: &str, pre_alloc: usize, post_alloc: usize) -> Result<Vec<u8>> {
//     let encoding_type = get_encoding_type(encoding);
//     let asn1_encoding = instance.get_encoding_with_type(&encoding_type);
//     let length = asn1_encoding.get_length();
//     let mut result = vec![0u8; pre_alloc + length + post_alloc];
//     result.resize(pre_alloc, 0);
//     let mut asn1_writer = Asn1Write::create_with_encoding(&mut result, encoding);
//     let writted_length = asn1_encoding.encode(&mut asn1_writer)?;
//     debug_assert_eq!(writted_length, length);
//     Ok(result)
// }

//pub struct Asn1ObjectImpl<'a> {

//}

// use super::asn1_encodable::{Asn1EncodableInternal, BER};

// pub(crate) trait Asn1ObjectInternal: Asn1EncodableInternal {
//     fn get_encoding_with_type(&self, encoding: &EncodingType) -> Box<dyn Asn1Encoding>;
// }

// pub struct Asn1ObjectImpl<T: Asn1ConvertiableInternal> {
//     instance: Box<Asn1EncodableImpl<T>>,
// }

// impl<T: Asn1ConvertiableInternal> Asn1ObjectImpl<T> {
//     pub fn new(instance: Box<T>) -> Self {
//         Asn1ObjectImpl {
//             instance: Box::new(Asn1EncodableImpl::new(instance)),
//         }
//     }
// }

// impl<'a> Asn1EncodableInternal for Asn1ObjectImpl<'a> {
//     fn encode_to(&self, writer: &mut dyn Write) -> Result<usize> {
//         let mut asn1_writer = Asn1Write::create_with_encoding(writer, BER);
//         let encoding = self.parent.get_encoding_with_type(asn1_writer.get_encoding());
//         encoding.encode(&mut asn1_writer)
//     }
// }

pub(crate) trait Asn1ObjectInternal {
    //fn as_asn1_encodable_internal(&self) -> &dyn Asn1EncodableInternal;
    fn get_encoding_with_type(&self, encoding: &EncodingType) -> Box<dyn Asn1Encoding>;
}
pub struct Asn1ObjectImpl {
    instance: Rc<dyn Asn1ObjectInternal>,
}

impl Asn1ObjectImpl {
    pub(crate) fn new(instance: Rc<dyn Asn1ObjectInternal>) -> Self {
        Asn1ObjectImpl { instance }
    }
    pub(crate) fn get_encoded_alloc(
        &self,
        encoding: &str,
        pre_alloc: usize,
        post_alloc: usize,
    ) -> Result<Vec<u8>> {
        let encoding_type = get_encoding_type(encoding);
        let asn1_encoding = self.instance.get_encoding_with_type(&encoding_type);
        let length = asn1_encoding.get_length();
        let mut result = vec![0u8; pre_alloc + length + post_alloc];
        result.resize(pre_alloc, 0);
        let mut asn1_writer = Asn1Write::create_with_encoding(&mut result, encoding);
        let writted_length = asn1_encoding.encode(&mut asn1_writer)?;
        debug_assert_eq!(writted_length, length);
        Ok(result)
    }
    // pub(crate) fn get_encoded(&self) -> Result<Vec<u8>> {
    //     Asn1EncodableImpl::new(self.instance.as_asn1_encodable_internal()).get_encoded()
    // }
}

impl Asn1Encodable for Asn1ObjectImpl {
    fn get_encoded(&self) -> Result<Vec<u8>> {
        self.get_encoded_alloc(BER, 0, 0)
    }
    
    fn get_encoded_with_encoding(&self, encoding: &str) -> Result<Vec<u8>> {
        self.get_encoded_alloc(encoding, 0, 0)
    }
    fn encode_to(&self, writer: &mut dyn Write) -> Result<usize> {
        let mut asn1_writer = Asn1Write::create_with_encoding(writer, BER);
        self.instance.get_encoding_with_type(asn1_writer.get_encoding()).encode(&mut asn1_writer)
    }
    fn encode_to_with_encoding(&self, writer: &mut dyn Write, encoding: &str) -> Result<usize> {
        let mut asn1_writer = Asn1Write::create_with_encoding(writer, encoding);
        self.instance.get_encoding_with_type(asn1_writer.get_encoding()).encode(&mut asn1_writer)
    }
}