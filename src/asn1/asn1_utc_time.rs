use crate::Result;

#[derive(Clone, Debug)]
pub struct Asn1UtcTime {
}

impl Asn1UtcTime {
    pub(crate) fn create_primitive(p0: Vec<u8>) -> Result<Self> {
        Ok(Asn1UtcTime {})
    }
}