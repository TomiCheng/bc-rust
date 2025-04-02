use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_tags::CONSTRUCTED;
use crate::asn1::asn1_write::get_length_of_encoding_il;
use crate::asn1::Asn1Write;
use crate::Result;

type ContentType = Box<dyn Asn1Encoding>;

pub(crate) struct TaggedILEncoding {
    tag_class: u32,
    tag_no: u32,
    contents_element: ContentType,
}

impl TaggedILEncoding {
    pub(crate) fn new(
        tag_class: u32,
        tag_no: u32,
        contents_element: ContentType,
    ) -> Self {
        TaggedILEncoding {
            tag_class,
            tag_no,
            contents_element,
        }
    }
}

impl Asn1Encoding for TaggedILEncoding {
    fn encode(&self, writer: &mut Asn1Write) -> Result<usize> {
        let mut length = 0;
        length += writer.write_identifier(CONSTRUCTED | self.tag_class, self.tag_no)?;
        length += writer.write_u8(0x80)?;
        length += self.contents_element.encode(writer)?;
        length += writer.write_u8(0x00)?;
        length += writer.write_u8(0x00)?;
        Ok(length)
    }

    fn get_length(&self) -> usize {
        get_length_of_encoding_il(self.tag_no, self.contents_element.as_ref())
    }
}
