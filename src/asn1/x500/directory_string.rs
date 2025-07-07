use crate::asn1::asn1_utilities::try_from_choice_tagged;
use crate::asn1::try_from_tagged::TryFromTagged;
use crate::asn1::{Asn1BmpString, Asn1Object, Asn1PrintableString, Asn1String, Asn1T61String, Asn1TaggedObject, Asn1UniversalString, Asn1Utf8String};
use crate::{BcError, Result};

/// ```text
/// DirectoryString ::= CHOICE {
///     teletexString TeletexString (SIZE (1..MAX)),
///     printableString PrintableString (SIZE (1..MAX)),
///     universalString UniversalString (SIZE (1..MAX)),
///     utf8String UTF8String (SIZE (1..MAX)),
///     bmpString BmpString (SIZE (1..MAX)),
/// }
/// ```
#[derive(Debug)]
pub enum DirectoryString {
    TeletexString(Asn1T61String),
    PrintableString(Asn1PrintableString),
    UniversalString(Asn1UniversalString),
    Utf8String(Asn1Utf8String),
    BmpString(Asn1BmpString),
}
impl DirectoryString {
    pub fn with_str(s: &str) -> Self {
        DirectoryString::Utf8String(Asn1Utf8String::with_str(s))
    }
}
impl TryFrom<Asn1Object> for DirectoryString {
    type Error = BcError;

    fn try_from(value: Asn1Object) -> Result<Self> {
        match value {
            Asn1Object::T61String(s) => Ok(DirectoryString::TeletexString(s)),
            Asn1Object::PrintableString(s) => Ok(DirectoryString::PrintableString(s)),
            Asn1Object::UniversalString(s) => Ok(DirectoryString::UniversalString(s)),
            Asn1Object::Utf8String(s) => Ok(DirectoryString::Utf8String(s)),
            Asn1Object::BmpString(s) => Ok(DirectoryString::BmpString(s)),
            _ => Err(BcError::with_invalid_format("Invalid DirectoryString type")),
        }
    }
}
impl From<DirectoryString> for Asn1Object {
    fn from(value: DirectoryString) -> Self {
        match value {
            DirectoryString::TeletexString(s) => Asn1Object::T61String(s),
            DirectoryString::PrintableString(s) => Asn1Object::PrintableString(s),
            DirectoryString::UniversalString(s) => Asn1Object::UniversalString(s),
            DirectoryString::Utf8String(s) => Asn1Object::Utf8String(s),
            DirectoryString::BmpString(s) => Asn1Object::BmpString(s),
        }
    }
}
impl TryFromTagged for DirectoryString {
    fn try_from_tagged(tagged: Asn1TaggedObject, declared_explicit: bool) -> Result<Self>
    where
        Self: Sized,
    {
        try_from_choice_tagged(tagged, declared_explicit, DirectoryString::try_from)
    }
}
impl Asn1String for DirectoryString {
    fn to_asn1_string(&self) -> Result<String> {
        match self {
            DirectoryString::TeletexString(s) => s.to_asn1_string(),
            DirectoryString::PrintableString(s) => s.to_asn1_string(),
            DirectoryString::UniversalString(s) => s.to_asn1_string(),
            DirectoryString::Utf8String(s) => s.to_asn1_string(),
            DirectoryString::BmpString(s) => s.to_asn1_string(),
        }
    }
}
