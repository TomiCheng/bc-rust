use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_write::{get_length_of_contents, get_length_of_encoding_dl};
use crate::asn1::{Asn1Write, asn1_tags};

pub(crate) struct ConstructedDlEncoding {
    tag_class: u8,
    tag_no: u8,
    encodings: Vec<Box<dyn Asn1Encoding>>,
    contents_length: usize,
}

impl ConstructedDlEncoding {
    pub(crate) fn new(tag_class: u8, tag_no: u8, encodings: Vec<Box<dyn Asn1Encoding>>) -> Self {
        let contents_length = get_length_of_contents(&encodings);
        ConstructedDlEncoding {
            tag_class,
            tag_no,
            encodings,
            contents_length,
        }
    }
}

impl Asn1Encoding for ConstructedDlEncoding {
    fn encode(&self, writer: &mut Asn1Write) -> crate::Result<usize> {
        let mut written = 0;
        written += writer.write_identifier(asn1_tags::CONSTRUCTED | self.tag_class, self.tag_no)?;
        written += writer.write_dl(self.contents_length)?;
        written += writer.write_encodings(&self.encodings)?;
        Ok(written)
    }
    fn get_length(&self) -> usize {
        get_length_of_encoding_dl(self.tag_no, self.contents_length)
    }
}
