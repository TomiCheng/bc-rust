use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_write::get_length_of_encoding_dl;
use crate::asn1::Asn1Write;
use crate::Result;

pub(crate) struct PrimitiveEncoding {
    tag_class: u8,
    tag_no: u8,
    contents_octets: Vec<u8>,
}

impl PrimitiveEncoding {
    pub fn new(tag_class: u8, tag_no: u8, contents_octets: Vec<u8>) -> Self {
        PrimitiveEncoding {
            tag_class,
            tag_no,
            contents_octets,
        }
    }
}

impl Asn1Encoding for PrimitiveEncoding {
    fn encode(&self, writer: &mut Asn1Write) -> Result<usize> {
        let mut length = 0;
        length += writer.write_identifier(self.tag_class, self.tag_no)?;
        length += writer.write_dl(self.contents_octets.len())?;
        length += writer.write(&self.contents_octets)?;
        Ok(length)
    }

    fn get_length(&self) -> usize {
        get_length_of_encoding_dl(self.tag_no, self.contents_octets.len())
    }
}