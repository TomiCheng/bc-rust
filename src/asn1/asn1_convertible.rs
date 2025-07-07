use crate::asn1::{Asn1Object, Asn1TaggedObject};
use crate::Result;

pub trait Asn1Convertible<T = Self> {
    fn to_asn1_object(&self) -> Result<Asn1Object>;

    fn from_tagged(tagged_object: Asn1TaggedObject, declared_explicit: bool) -> Result<T> {
        todo!();
    }

    fn from_optional(asn1_object: Asn1Object) -> Result<Option<T>> {
        todo!();
    }

    fn from_asn1_object(asn1_object: Asn1Object) -> Result<T> {
        if let Some(result) = Self::from_optional(asn1_object)? {
            Ok(result)
        } else {
            Err(crate::BcError::with_invalid_operation("Expected an ASN.1 object"))
        }
    }
}