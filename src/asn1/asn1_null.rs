use crate::Result;
use std::fmt;

/// A Null object.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Asn1Null;

impl Asn1Null {
    pub fn new() -> Self {
        Asn1Null
    }
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        if !contents.is_empty() {
            return Err(crate::BcError::with_invalid_operation("malformed NULL encoding encountered"));
        }
        Ok(Asn1Null)
    }
}
impl fmt::Display for Asn1Null {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NULL")
    }
}
