use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::{asn1_tags, Asn1Write};
use crate::asn1::asn1_write::get_length_of_encodings_il;
use crate::Result;
pub(crate) struct ConstructedIlEncoding {
    tag_class: u8,
    tag_no: u8,
    encodings: Vec<Box<dyn Asn1Encoding>>,
}

impl ConstructedIlEncoding {
    pub(crate) fn new(tag_class: u8, tag_no: u8, encodings: Vec<Box<dyn Asn1Encoding>>) -> Self {
        ConstructedIlEncoding {
            tag_class,
            tag_no,
            encodings,
        }
    }
}
impl Asn1Encoding for ConstructedIlEncoding {
    fn encode(&self, writer: &mut Asn1Write) -> Result<usize> {
        let mut written = 0;
        written += writer.write_identifier(asn1_tags::CONSTRUCTED | self.tag_class, self.tag_no)?;
        written += writer.write_u8(0x80)?;
        written += writer.write_encodings(&self.encodings)?;
        written += writer.write_u8(0x00)?;
        written += writer.write_u8(0x00)?;
        Ok(written)
    }

    fn get_length(&self) -> usize {
        get_length_of_encodings_il(self.tag_no, &self.encodings)
    }
}