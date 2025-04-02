use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_tags::CONSTRUCTED;
use crate::asn1::asn1_write::get_length_of_encodings_il;
use crate::asn1::Asn1Write;
use crate::Result;

type ContentType = Box<dyn Asn1Encoding>;

pub(crate) struct ConstructedILEncoding {
    tag_class: u32,
    tag_no: u32,
    contents_elements: Vec<ContentType>,
}

impl ConstructedILEncoding {
    pub(crate) fn new(tag_class: u32, tag_no: u32, contents_elements: Vec<ContentType>) -> Self {
        ConstructedILEncoding {
            tag_class,
            tag_no,
            contents_elements,
        }
    }
}

impl Asn1Encoding for ConstructedILEncoding {
    fn encode(&self, writer: &mut Asn1Write) -> Result<usize> {
        let mut length = 0;
        length += writer.write_identifier(CONSTRUCTED | self.tag_class, self.tag_no)?;
        length += writer.write_u8(0x80)?;
        length += writer.encode_contents(&self.contents_elements)?;
        length += writer.write_u8(0x00)?;
        length += writer.write_u8(0x00)?;
        Ok(length)
    }

    fn get_length(&self) -> usize {
        get_length_of_encodings_il(self.tag_no, &self.contents_elements)
    }
}
