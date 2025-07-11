use crate::Result;
use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::primitive_encoding::PrimitiveEncoding;
use crate::asn1::{Asn1String, EncodingType, asn1_tags};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Asn1PrintableString {
    contents: String,
}
impl Asn1PrintableString {
    fn new(contents: String) -> Self {
        Asn1PrintableString { contents }
    }
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        let s = String::from_utf8(contents)?;
        // do not validate whether the content is a printable string, as this may cause errors.
        Ok(Asn1PrintableString::new(s))
    }
    pub fn with_str(s: &str) -> Result<Self> {
        if !Self::is_printable_string(s) {
            return Err(crate::BcError::with_invalid_argument("Invalid PrintableString content"));
        }
        Ok(Asn1PrintableString::new(s.to_string()))
    }
    pub fn with_str_validate(s: &str, validate: bool) -> Result<Self> {
        if validate && !Self::is_printable_string(s) {
            return Err(crate::BcError::with_invalid_argument("Invalid PrintableString content"));
        }
        Ok(Asn1PrintableString::new(s.to_string()))
    }
    pub fn is_printable_string(s: &str) -> bool {
        s.chars().all(|c| {
            c.is_ascii()
                && (c.is_alphanumeric()
                    || c == ' '
                    || c == '\''
                    || c == '('
                    || c == ')'
                    || c == '+'
                    || c == '-'
                    || c == '.'
                    || c == ':'
                    || c == '='
                    || c == '?'
                    || c == '/'
                    || c == ',')
        })
    }
}
impl Asn1String for Asn1PrintableString {
    fn to_asn1_string(&self) -> Result<String> {
        Ok(self.contents.clone())
    }
}
impl fmt::Display for Asn1PrintableString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.contents)
    }
}
impl Asn1EncodingInternal for Asn1PrintableString {
    fn get_encoding(&self, encoding_type: EncodingType) -> Box<dyn Asn1Encoding> {
        self.get_encoding_implicit(encoding_type, asn1_tags::UNIVERSAL, asn1_tags::PRINTABLE_STRING)
    }
    fn get_encoding_implicit(&self, encoding_type: EncodingType, tag_class: u8, tag_no: u8) -> Box<dyn Asn1Encoding> {
        Box::new(PrimitiveEncoding::new(
            tag_class,
            tag_no,
            self.contents.as_bytes().to_vec(),
        ))
    }
}
