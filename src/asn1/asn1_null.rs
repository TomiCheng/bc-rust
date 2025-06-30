use std::fmt::{Display, Formatter};
use crate::Result;

#[derive(Clone, Debug)]
pub struct Asn1Null;

impl Asn1Null {
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        if !contents.is_empty() {
            return Err(crate::BcError::with_invalid_operation("malformed NULL encoding encountered"));
        }
        Ok(Asn1Null)
    }
}

impl Display for Asn1Null {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("NULL")
    }
}