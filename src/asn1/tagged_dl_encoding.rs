use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_tags::CONSTRUCTED;
use crate::asn1::asn1_write::get_length_of_encoding_dl;
use crate::asn1::Asn1Write;
use crate::Result;

type ContentType = Box<dyn Asn1Encoding>;

pub(crate) struct TaggedDLEncoding {
    tag_class: u32,
    tag_no: u32,
    contents_element: ContentType,
    contents_length: usize,
}

impl TaggedDLEncoding {
    pub(crate) fn new(tag_class: u32, tag_no: u32, contents_element: ContentType) -> Self {
        let contents_length = contents_element.get_length();
        TaggedDLEncoding {
            tag_class,
            tag_no,
            contents_element,
            contents_length,
        }
    }
}

impl Asn1Encoding for TaggedDLEncoding {
    fn encode(&self, writer: &mut Asn1Write) -> Result<usize> {
        let mut length = 0;
        length += writer.write_identifier(CONSTRUCTED | self.tag_class, self.tag_no)?;
        length += writer.write_dl(self.contents_length as u32)?;
        length += self.contents_element.encode(writer)?;
        Ok(length)
    }

    fn get_length(&self) -> usize {
        get_length_of_encoding_dl(self.tag_no, self.contents_length)
    }
}
