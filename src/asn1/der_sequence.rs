// use std::fmt::{Display, Formatter, Debug};
// use std::io::Write;
// use std::rc::Rc;

// use super::asn1_convertiable::Asn1Convertiable;
// use super::asn1_object::Asn1ObjectImpl;
// use super::Asn1Encodable;
// use crate::Result;

// #[derive(Clone)]
// pub struct DerSequenceImpl {
//     elements: Rc<Vec<Box<dyn Asn1Convertiable>>>,
// }

// impl DerSequenceImpl {
//     pub fn new(elements: Rc<Vec<Box<dyn Asn1Convertiable>>>) -> Self {
//         DerSequenceImpl {
//             elements
//         }
//     }
// }

// impl Asn1ObjectImpl for DerSequenceImpl {}
// impl Display for DerSequenceImpl {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         todo!();
//     }
// }
// impl Debug for DerSequenceImpl {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         //f.debug_struct("DerSequenceImpl").field("elements", &self.elements).finish()
//         todo!();
//     }
// }
// impl Asn1Encodable for DerSequenceImpl {
//     fn encode_to_with_encoding(&self, writer: &mut dyn Write, encoding: &str) -> Result<usize> {
//         todo!()
//     }

//     fn get_encoded_with_encoding(&self, encoding: &str) -> Result<Vec<u8>> {
//         todo!()
//     }
// }
