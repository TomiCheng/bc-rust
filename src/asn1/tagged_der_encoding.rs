use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_tags;
use crate::asn1::asn1_write;
use crate::asn1::der_encoding::DerEncoding;
use crate::asn1::Asn1Write;
use crate::Result;

pub struct TaggedDerEncoding {
    tag_class: u32,
    tag_no: u32,
    contents_element: Box<dyn DerEncoding>,
    contents_length: usize,
}

impl TaggedDerEncoding {
    pub(crate) fn new(tag_class: u32, tag_no: u32, contents_element: Box<dyn DerEncoding>) -> Self {
        let length = contents_element.get_length();
        TaggedDerEncoding {
            tag_class,
            tag_no,
            contents_element,
            contents_length: length,
        }
    }
}

impl DerEncoding for TaggedDerEncoding {}
impl Asn1Encoding for TaggedDerEncoding {
    fn encode(&self, writer: &mut Asn1Write) -> Result<usize> {
        let mut length = 0;
        length += writer.write_identifier(asn1_tags::CONSTRUCTED | self.tag_class, self.tag_no)?;
        length += writer.write_dl(self.contents_length as u32)?;
        length += self.contents_element.encode(writer)?;
        Ok(length)
    }

    fn get_length(&self) -> usize {
        asn1_write::get_length_of_encoding_dl(self.tag_no, self.contents_length)
    }
}
