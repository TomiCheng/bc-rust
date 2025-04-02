use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_tags::CONSTRUCTED;
use crate::asn1::asn1_write::{get_length_of_contents, get_length_of_encoding_dl};
use crate::asn1::Asn1Write;

pub(crate) struct ConstructedDerEncoding {
    tag_class: u32,
    tag_no: u32,
    contents_elements: Vec<Box<dyn Asn1Encoding>>,
    contents_length: usize,
}

impl ConstructedDerEncoding {
    pub(crate) fn new(
        tag_class: u32,
        tag_no: u32,
        contents_elements: Vec<Box<dyn Asn1Encoding>>,
    ) -> Self {
        let contents_length = get_length_of_contents(&contents_elements);
        ConstructedDerEncoding {
            tag_class,
            tag_no,
            contents_elements,
            contents_length,
        }
    }
}

impl Asn1Encoding for ConstructedDerEncoding {
    fn encode(&self, writer: &mut Asn1Write) -> crate::Result<usize> {
        let mut length = 0;
        length += writer.write_identifier(CONSTRUCTED | self.tag_class, self.tag_no)?;
        length += writer.write_dl(self.contents_length as u32)?;
        length += writer.encode_contents(&self.contents_elements)?;
        Ok(length)
    }

    fn get_length(&self) -> usize {
        get_length_of_encoding_dl(self.tag_no, self.contents_length)
    }
}
