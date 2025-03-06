use std::fmt::{Display, Formatter};
use std::io::Write;
use std::rc::Rc;

use super::asn1_encoding::Asn1Encoding;
use super::asn1_object::{encode_to_with_encoding, get_encoded_with_encoding, Asn1ObjectImpl};
use super::asn1_tags::{OCTET_STRING, UNIVERSAL};
use super::asn1_write::{get_encoding_type, EncodingType};
use super::primitive_encoding::PrimitiveEncoding;
use super::Asn1Encodable;
use crate::util::encoders::hex::to_hex_string;
use crate::Result;

pub struct DerOctetStringImpl {
    contents: Rc<Vec<u8>>,
}

impl DerOctetStringImpl {
    pub fn new(contents: Rc<Vec<u8>>) -> Self {
        DerOctetStringImpl { contents }
    }
    fn get_encoding_with_type(&self, _encode_type: EncodingType) -> Box<dyn Asn1Encoding> {
        Box::new(PrimitiveEncoding::new(
            UNIVERSAL,
            OCTET_STRING,
            self.contents.clone(),
        ))
    }
}

impl Asn1ObjectImpl for DerOctetStringImpl {}
impl Display for DerOctetStringImpl {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", to_hex_string(&self.contents))
    }
}
impl Asn1Encodable for DerOctetStringImpl {
    fn get_encoded_with_encoding(&self, encoding_str: &str) -> Result<Vec<u8>> {
        let encoding = self.get_encoding_with_type(get_encoding_type(encoding_str));
        get_encoded_with_encoding(encoding_str, encoding.as_ref())
    }

    fn encode_to_with_encoding(&self, writer: &mut dyn Write, encoding_str: &str) -> Result<usize> {
        let asn1_encoding = self.get_encoding_with_type(get_encoding_type(encoding_str));
        encode_to_with_encoding(writer, encoding_str, asn1_encoding.as_ref())
    }
}
