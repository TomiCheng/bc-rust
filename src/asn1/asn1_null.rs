use std::fmt::{Display, Formatter};
use crate::asn1::{Asn1Convertible, Asn1Object, Asn1TaggedObject};
use crate::Result;

/// A Null object.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
        write!(f, "NULL")
    }
}

impl Asn1Convertible for Asn1Null {
    fn to_asn1_object(&self) -> Result<Asn1Object> {
        Ok(Asn1Object::Null(Asn1Null))
    }

    fn from_tagged(tagged_object: Asn1TaggedObject, declared_explicit: bool) -> Result<Self> {
        
        todo!()
    }
}