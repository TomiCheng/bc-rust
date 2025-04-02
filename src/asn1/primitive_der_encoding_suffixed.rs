use super::Asn1Write;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::der_encoding::DerEncoding;

pub(crate) struct PrimitiveDerEncodingSuffixed {
    tag_class: u32,
    tag_no: u32,
    contents_octets: Vec<u8>,
    contents_suffix: u8,
}

impl PrimitiveDerEncodingSuffixed {
    pub(crate) fn new(
        tag_class: u32,
        tag_no: u32,
        contents_octets: &[u8],
        contents_suffix: u8,
    ) -> Self {
        assert!(contents_octets.len() > 0);

        PrimitiveDerEncodingSuffixed {
            tag_class,
            tag_no,
            contents_octets: contents_octets.to_vec(),
            contents_suffix,
        }
    }
}

impl DerEncoding for PrimitiveDerEncodingSuffixed {}

impl Asn1Encoding for PrimitiveDerEncodingSuffixed {
    fn encode(&self, writer: &mut Asn1Write) -> crate::Result<usize> {
        todo!()
    }

    fn get_length(&self) -> usize {
        todo!()
    }
}
