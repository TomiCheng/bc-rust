use crate::asn1::{Asn1BmpString, Asn1PrintableString, Asn1T61String, Asn1TaggedObject, Asn1UniversalString, Asn1Utf8String};
use crate::Result;

/// ```text
/// DirectoryString ::= CHOICE {
///     teletexString TeletexString (SIZE (1..MAX)),
///     printableString PrintableString (SIZE (1..MAX)),
///     universalString UniversalString (SIZE (1..MAX)),
///     utf8String UTF8String (SIZE (1..MAX)),
///     bmpString BmpString (SIZE (1..MAX)),
/// }
/// ```
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
    pub fn get_tagged(tagged_object: Asn1TaggedObject, declared_explicit: bool) -> Result<Option<Self>> {
        todo!();
    }
}