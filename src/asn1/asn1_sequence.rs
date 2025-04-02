use std::fmt;
use std::any;
use std::sync;
use std::ops;

use super::*;
use crate::Result;

#[derive(Debug, Clone)]
pub struct Asn1Sequence {
    elements: sync::Arc<Vec<sync::Arc<dyn Asn1Encodable>>>,
}

impl Asn1Sequence {
    pub fn with_asn1_encodables(elements: Vec<sync::Arc<dyn Asn1Encodable>>) -> Self {
        Asn1Sequence {
            elements: sync::Arc::new(elements),
        }
    }

    pub fn iter(&self) -> &[sync::Arc<dyn Asn1Encodable>] {
        self.elements.as_slice()
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }
}

// trait
impl Asn1Object for Asn1Sequence {}
impl Asn1Encodable for Asn1Sequence {
    fn encode_to_with_encoding(&self, writer: &mut dyn std::io::Write, encoding: &str) -> Result<usize> {
        todo!();
        // let mut written = 0;
        // for element in self.elements.iter() {
        //     written += element.encode_to_with_encoding(writer, encoding)?;
        // }
        // Ok(written)
    }

    fn get_encoded_with_encoding(&self, encoding: &str) -> Result<Vec<u8>> {
        todo!();
        // let mut encoded = Vec::new();
        // for element in self.elements.iter() {
        //     encoded.extend(element.get_encoded_with_encoding(encoding)?);
        // }
        // Ok(encoded)
    }
}
impl Asn1Convertiable for Asn1Sequence {
    fn to_asn1_object(&self) -> sync::Arc<dyn Asn1Object> {
        sync::Arc::new(self.clone())
    }
    fn as_any(&self) -> sync::Arc<dyn any::Any> {
        sync::Arc::new(self.clone())
    }
}

impl ops::Index<usize> for Asn1Sequence {
    type Output = sync::Arc<dyn Asn1Encodable>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.elements[index]
    }
}

impl fmt::Display for Asn1Sequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SEQUENCE")
    }
}