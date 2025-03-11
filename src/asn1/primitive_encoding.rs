use std::sync;

use super::asn1_encoding::Asn1Encoding;
use super::asn1_write::get_length_of_encoding_dl;
use super::Asn1Write;

pub(crate) struct PrimitiveEncoding {
    tag_class: u32,
    tag_no: u32,
    contents_octets: sync::Arc<Vec<u8>>,
}

impl PrimitiveEncoding {
    pub fn new(tag_class: u32, tag_no: u32, contents_octets: sync::Arc<Vec<u8>>) -> Self {
        PrimitiveEncoding {
            tag_class,
            tag_no,
            contents_octets: contents_octets,
        }
    }
}

impl Asn1Encoding for PrimitiveEncoding {
    fn encode(&self, writer: &mut Asn1Write) -> crate::Result<usize> {
        let mut length = 0;
        length += writer.write_identifier(self.tag_class, self.tag_no)?;
        length += writer.write_dl(self.contents_octets.len() as u32)?;
        length += writer.write(&self.contents_octets)?;
        Ok(length)
    }

    fn get_length(&self) -> usize {
        get_length_of_encoding_dl(self.tag_no, self.contents_octets.len())
    }
}
