use crate::Result;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_write::get_length_of_encoding_dl;
use crate::asn1::{Asn1Write, asn1_tags};

pub(crate) struct TaggedDerEncoding {
    tag_class: u8,
    tag_no: u8,
    base_encoding: Box<dyn Asn1Encoding>,
    contents_length: usize,
}

impl TaggedDerEncoding {
    pub fn new(tag_class: u8, tag_no: u8, base_encoding: Box<dyn Asn1Encoding>) -> Self {
        let contents_length = base_encoding.get_length();
        TaggedDerEncoding {
            tag_class,
            tag_no,
            base_encoding,
            contents_length,
        }
    }
}
impl Asn1Encoding for TaggedDerEncoding {
    fn encode(&self, writer: &mut Asn1Write) -> Result<usize> {
        let mut length = 0;
        length += writer.write_identifier(asn1_tags::CONSTRUCTED | self.tag_class, self.tag_no)?;
        length += writer.write_dl(self.contents_length)?;
        length += self.base_encoding.encode(writer)?;
        Ok(length)
    }

    fn get_length(&self) -> usize {
        get_length_of_encoding_dl(self.tag_no, self.contents_length)
    }
}
