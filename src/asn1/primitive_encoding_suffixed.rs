use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::Asn1Write;

pub(crate) struct PrimitiveEncodingSuffixed {
    tag_class: u8,
    tag_no: u8,
    contents_octets: Vec<u8>,
    contents_suffix: u8,
}

impl PrimitiveEncodingSuffixed {
    pub fn new(tag_class: u8, tag_no: u8, contents_octets: Vec<u8>, contents_suffix: u8) -> Self {
        PrimitiveEncodingSuffixed {
            tag_class,
            tag_no,
            contents_octets,
            contents_suffix,
        }
    }
}

impl Asn1Encoding for PrimitiveEncodingSuffixed {
    fn encode(&self, writer: &mut Asn1Write) -> crate::Result<usize> {
        let mut length = 0;
        length += writer.write_identifier(self.tag_class, self.tag_no)?;
        length += writer.write_dl(self.contents_octets.len())?;
        length += writer.write(&self.contents_octets[..(&self.contents_octets.len() - 1)])?;
        length += writer.write_u8(self.contents_suffix)?;
        Ok(length)
    }
}