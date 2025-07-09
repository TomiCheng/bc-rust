use crate::asn1::Asn1String;
use crate::Result;

#[derive(Clone, Debug)]
pub struct Asn1PrintableString {
    contents: String,
}

impl Asn1PrintableString {
    fn new(contents: String) -> Self {
        Asn1PrintableString { contents }
    }
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        let s = String::from_utf8(contents)?;
        if !Self::is_printable_string(&s) {
            return Err(crate::BcError::with_invalid_argument("Invalid PrintableString content"));
        }
        Ok(Asn1PrintableString::new(s))
    }
    pub fn with_str(s: &str) -> Result<Self> {
        if !Self::is_printable_string(s) {
            return Err(crate::BcError::with_invalid_argument("Invalid PrintableString content"));
        }
        Ok(Asn1PrintableString::new(s.to_string()))
    }
    pub fn is_printable_string(s: &str) -> bool {
        s.chars().all(|c| c.is_ascii() && (c.is_alphanumeric() || c.is_whitespace() ||
            c == '\'' ||
            c == '(' ||
            c == ')' ||
            c == '+' ||
            c == '-' ||
            c == '.' ||
            c == ':' ||
            c == '=' ||
            c == '?' ||
            c == '/' ||
            c == ','
        ))
    }
}

impl Asn1String for Asn1PrintableString {
    fn to_asn1_string(&self) -> Result<String> {
        Ok(self.contents.clone())
    }
}