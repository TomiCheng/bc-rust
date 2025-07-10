use crate::BcError;
use crate::asn1::Asn1TaggedObject;

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

pub(crate) trait TryIntoTagged<T>: Sized {
    fn try_into_tagged(self, declared_explicit: bool) -> Result<T, BcError>;
}

impl<T> TryIntoTagged<T> for Asn1TaggedObject
where
    T: TryFromTagged,
{
    fn try_into_tagged(self, declared_explicit: bool) -> Result<T, BcError> {
        T::try_from_tagged(self, declared_explicit)
    }
}
