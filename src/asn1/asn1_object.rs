use std::any::Any;
use std::io::{Read, Write};
use std::rc::Rc;

use super::asn1_encodable::BER;
use super::asn1_encoding::Asn1Encoding;
use super::asn1_write::EncodingType;
use super::{Asn1Encodable, Asn1Read, Asn1Write};
use crate::asn1::asn1_write::get_encoding_type;
use crate::Result;

pub(crate) trait Asn1ObjectInternal {
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
        self.instance
            .get_encoding_with_type(asn1_writer.get_encoding())
            .encode(&mut asn1_writer)
    }
    fn encode_to_with_encoding(&self, writer: &mut dyn Write, encoding: &str) -> Result<usize> {
        let mut asn1_writer = Asn1Write::create_with_encoding(writer, encoding);
        self.instance
            .get_encoding_with_type(asn1_writer.get_encoding())
            .encode(&mut asn1_writer)
    }
}

/// Read a base ASN.1 object from a Read.
/// # Arguments
/// * `reader` - The Read to parse.
/// # Returns
/// The base ASN.1 object represented by the byte array.
/// # Errors
/// If there is a problem parsing the data.
pub fn parse_asn1_object(reader: &mut dyn Read) -> Result<Box<dyn Any>> {
    let mut asn1_reader = Asn1Read::new(reader, i32::MAX as usize);
    asn1_reader.read_object()
}
