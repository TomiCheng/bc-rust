use std::fmt::Display;

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
