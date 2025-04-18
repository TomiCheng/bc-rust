use super::Asn1Write;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_write::get_length_of_encoding_dl;
use crate::Result;
use std::rc::Rc;

pub(crate) struct PrimitiveEncodingSuffixed {
    tag_class: u32,
    tag_no: u32,
    contents_octets: Rc<Vec<u8>>,
    contents_suffix: u8,
}

impl PrimitiveEncodingSuffixed {
    pub(crate) fn new(
        tag_class: u32,
        tag_no: u32,
        contents_octets: Rc<Vec<u8>>,
        contents_suffix: u8,
    ) -> Self {
        PrimitiveEncodingSuffixed {
            tag_class,
            tag_no,
            contents_octets,
            contents_suffix,
        }
    }
}

impl Asn1Encoding for PrimitiveEncodingSuffixed {
    fn encode(&self, writer: &mut Asn1Write) -> Result<usize> {
        let mut length = 0;
        length += writer.write_identifier(self.tag_class, self.tag_no)?;
        length += writer.write_dl(self.contents_octets.len() as u32)?;
        length += writer.write(&self.contents_octets[..(&self.contents_octets.len() - 1)])?;
        length += writer.write_u8(self.contents_suffix)?;
        Ok(length)
    }

    fn get_length(&self) -> usize {
        get_length_of_encoding_dl(self.tag_no, self.contents_octets.len())
    }
}
