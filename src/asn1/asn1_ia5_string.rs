use crate::Result;
use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_universal_type::Asn1UniversalType;
use crate::asn1::primitive_encoding::PrimitiveEncoding;
use crate::asn1::try_from_tagged::TryFromTagged;
use crate::asn1::{Asn1Object, Asn1OctetString, Asn1String, Asn1TaggedObject, EncodingType, asn1_tags};

/// IA5String object - this is an Ascii string.
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct Asn1Ia5String {
    content: String,
}

impl Asn1Ia5String {
    pub fn new(contents: String) -> Result<Self> {
        if !Self::is_ia5_string(&contents) {
            return Err(crate::BcError::with_invalid_argument("Invalid IA5String content"));
        }
        Ok(Asn1Ia5String { content: contents })
    }
    pub fn with_str(s: &str) -> Result<Self> {
        Asn1Ia5String::new(s.to_string())
    }
    pub fn is_ia5_string(s: &str) -> bool {
        s.chars().all(|c| c.is_ascii())
    }
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        let s = String::from_utf8(contents)?;
        Asn1Ia5String::new(s)
    }
}

impl Asn1String for Asn1Ia5String {
    fn to_asn1_string(&self) -> Result<String> {
        Ok(self.content.clone())
    }
}
impl TryFromTagged for Asn1Ia5String {
    fn try_from_tagged(tagged: Asn1TaggedObject, declared_explicit: bool) -> Result<Self>
    where
        Self: Sized,
    {
        tagged.try_from_base_universal(declared_explicit, Asn1Ia5StringMetadata)
    }
}
impl Asn1EncodingInternal for Asn1Ia5String {
    fn get_encoding(&self, _: EncodingType) -> Box<dyn Asn1Encoding> {
        Box::new(PrimitiveEncoding::new(
            asn1_tags::UNIVERSAL,
            asn1_tags::IA5_STRING,
            self.content.as_bytes().to_vec(),
        ))
    }
}
struct Asn1Ia5StringMetadata;
impl Asn1UniversalType<Asn1Ia5String> for Asn1Ia5StringMetadata {
    fn checked_cast(&self, asn1_object: Asn1Object) -> Result<Asn1Ia5String> {
        asn1_object.try_into()
    }
    fn implicit_primitive(&self, octets: Asn1OctetString) -> Result<Asn1Ia5String> {
        Asn1Ia5String::create_primitive(octets.into_vec())
    }
}
