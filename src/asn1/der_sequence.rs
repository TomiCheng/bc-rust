use std::fmt;
// use std::io::Write;
// use std::rc::Rc;

// use super::asn1_convertiable::Asn1Convertiable;
// use super::asn1_object::Asn1ObjectImpl;
// use super::Asn1Encodable;

use std::sync;

use super::*;
use crate::Result;

#[derive(Debug, Clone)]
pub struct DerSequence {
    parent: Asn1Sequence,
//     elements: Rc<Vec<Box<dyn Asn1Convertiable>>>,
}

impl DerSequence {
//     pub fn new(elements: Rc<Vec<Box<dyn Asn1Convertiable>>>) -> Self {
//         DerSequenceImpl {
//             elements
//         }
//     }
    pub fn with_asn1_encodables(elements: Vec<sync::Arc<dyn Asn1Encodable>>) -> Self {
        DerSequence {
            parent: Asn1Sequence::with_asn1_encodables(elements)
        }
    }
}

// trait
impl Asn1Object for DerSequence {}
impl Asn1Encodable for DerSequence {
    fn encode_to_with_encoding(&self, writer: &mut dyn std::io::Write, encoding: &str) -> Result<usize> {
        todo!();
    }

    fn get_encoded_with_encoding(&self, encoding: &str) -> Result<Vec<u8>> {
        todo!();
    }
}
impl Asn1Convertiable for DerSequence {
    fn to_asn1_object(&self) -> sync::Arc<dyn Asn1Object> {
        sync::Arc::new(self.clone())
    }
    fn as_any(&self) -> sync::Arc<dyn std::any::Any> {
        sync::Arc::new(self.clone())
    }
}

// impl Asn1ObjectImpl for DerSequenceImpl {}
impl fmt::Display for DerSequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        todo!();
    }
}
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
