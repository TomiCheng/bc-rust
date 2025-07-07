use crate::asn1::Asn1TaggedObject;
use crate::BcError;

pub(crate) trait TryFromTagged {
    /// Attempts to convert a tagged ASN.1 object into a specific type.
    ///
    /// # Arguments
    /// * `tagged` - The tagged ASN.1 object to convert.
    /// * `declared_explicit` - A boolean indicating whether the tag is declared as explicit.
    ///
    /// # Returns
    /// * `Ok(T)` if the conversion is successful.
    /// * `Err(BcError)` if the conversion fails.
    fn try_from_tagged(tagged: Asn1TaggedObject, declared_explicit: bool) -> Result<Self, BcError>
    where
        Self: Sized;
}