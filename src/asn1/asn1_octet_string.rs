use std::fmt::Display;
use crate::Result;

#[derive(Clone, Debug)]
pub struct Asn1OctetString {
    contents: Vec<u8>,
}

impl Asn1OctetString {
    pub fn new(contents: Vec<u8>) -> Self {
        Self { contents }
    }
    pub(crate) fn with_contents(contents: &[u8]) -> Self {
        Self {
            contents: contents.to_vec(),
        }
    }
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        Ok(Self { contents })
    }
}

impl Display for Asn1OctetString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("#")?;
        for byte in &self.contents {
            write!(f, "{:02X}", byte)?;
        }
        Ok(())
    }
}
