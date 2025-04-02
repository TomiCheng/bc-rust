use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_write::get_length_of_encoding_dl;
use crate::asn1::der_encoding::DerEncoding;
use crate::asn1::Asn1Write;
use crate::Result;
use std::sync::Arc;

pub(crate) struct PrimitiveDerEncoding {
    tag_class: u32,
    tag_no: u32,
    contents_octets: Arc<Vec<u8>>,
}

impl PrimitiveDerEncoding {
    pub(crate) fn new(tag_class: u32, tag_no: u32, contents_octets: Arc<Vec<u8>>) -> Self {
        PrimitiveDerEncoding {
            tag_class,
            tag_no,
            contents_octets,
        }
    }
}

impl Asn1Encoding for PrimitiveDerEncoding {
    fn encode(&self, writer: &mut Asn1Write) -> Result<usize> {
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

impl DerEncoding for PrimitiveDerEncoding {}
