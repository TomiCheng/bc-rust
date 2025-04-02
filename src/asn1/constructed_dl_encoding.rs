use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_write::{get_length_of_contents, get_length_of_encoding_dl};
use crate::asn1::Asn1Write;
use crate::Result;

type ContentType = Box<dyn Asn1Encoding>;
pub(crate) struct ConstructedDLEncoding {
    tag_class: u32,
    tag_no: u32,
    contents_elements: Vec<ContentType>,
    contents_length: usize,
}

impl ConstructedDLEncoding {
    pub(crate) fn new(tag_class: u32, tag_no: u32, contents_elements: Vec<ContentType>) -> Self {
        let contents_length = get_length_of_contents(&contents_elements);
        ConstructedDLEncoding {
            tag_class,
            tag_no,
            contents_elements,
            contents_length,
        }
    }
}

impl Asn1Encoding for ConstructedDLEncoding {
    fn encode(&self, writer: &mut Asn1Write) -> Result<usize> {
        let mut length = 0;
        length += writer.write_identifier(self.tag_class, self.tag_no)?;
        length += writer.write_dl(self.contents_length as u32)?;
        length += writer.encode_contents(&self.contents_elements)?;
        Ok(length)
    }

    fn get_length(&self) -> usize {
        get_length_of_encoding_dl(self.tag_no, self.contents_length)
    }
}
