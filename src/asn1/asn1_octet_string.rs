use crate::asn1::asn1_universal_type::Asn1UniversalType;
use crate::asn1::try_from_tagged::TryFromTagged;
use crate::asn1::{asn1_tags, Asn1Object, Asn1TaggedObject, EncodingType};
use crate::{BcError, Result};
use std::fmt::Display;
use std::hash::{Hash};
use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::primitive_encoding::PrimitiveEncoding;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Asn1OctetString {
    contents: Vec<u8>,
}

impl Asn1OctetString {
    pub fn into_vec(self) -> Vec<u8> {
        self.contents
    }
}
impl Asn1OctetString {
    pub fn new(contents: Vec<u8>) -> Self {
        Self { contents }
    }
    pub(crate) fn from_asn1_object(asn1_object: Asn1Object) -> Result<Self> {
        if let Asn1Object::OctetString(octet_string) = asn1_object {
            Ok(octet_string)
        } else {
            Err(BcError::with_invalid_argument("not an Asn1OctetString object"))
        }
    }
    pub(crate) fn with_contents(contents: &[u8]) -> Self {
        Self { contents: contents.to_vec() }
    }
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        Ok(Self { contents })
    }
    pub fn get_octets(&self) -> &Vec<u8> {
        &self.contents
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
impl TryFromTagged for Asn1OctetString {
    fn try_from_tagged(tagged: Asn1TaggedObject, declared_explicit: bool) -> std::result::Result<Self, BcError>
    where
        Self: Sized,
    {
        tagged.try_from_base_universal(declared_explicit, Asn1OctetStringMetadata)
    }
}
impl Asn1EncodingInternal for Asn1OctetString {
    fn get_encoding(&self, encoding_type: EncodingType) -> Box<dyn Asn1Encoding> {
        self.get_encoding_implicit(encoding_type, asn1_tags::UNIVERSAL, asn1_tags::OCTET_STRING)
    }
    fn get_encoding_implicit(&self, _encoding_type: EncodingType, tag_class: u8, tag_no: u8) -> Box<dyn Asn1Encoding> {
        Box::new(PrimitiveEncoding::new(
            tag_class,
            tag_no,
            self.contents.clone(),
        ))
    }
}
struct Asn1OctetStringMetadata;

impl Asn1UniversalType<Asn1OctetString> for Asn1OctetStringMetadata {
    fn checked_cast(&self, asn1_object: Asn1Object) -> Result<Asn1OctetString> {
        asn1_object.try_into()
    }

    fn implicit_primitive(&self, value: Asn1OctetString) -> Result<Asn1OctetString> {
        Ok(value)
    }
}
