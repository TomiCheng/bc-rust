use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::Asn1Write;

pub(crate) struct TaggedEncoding {
    tag_class: u8,
    tag_no: u8,
    base_encoding: Box<dyn Asn1Encoding>,
}

impl TaggedEncoding {
    pub fn new(tag_class: u8, tag_no: u8, base_encoding: Box<dyn Asn1Encoding>) -> Self {
        TaggedEncoding {
            tag_class,
            tag_no,
            base_encoding,
        }
    }
}
impl Asn1Encoding for TaggedEncoding {
    fn encode(&self, writer: &mut Asn1Write) -> crate::Result<usize> {
        todo!()
    }

    fn get_length(&self) -> usize {
        todo!()
    }
}