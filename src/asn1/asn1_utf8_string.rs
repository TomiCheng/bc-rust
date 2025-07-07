use std::fmt::Display;

#[derive(Clone, Debug)]
pub struct Asn1Utf8String {
    content: String,
}

impl Asn1Utf8String {
    pub fn new(content: String) -> Self {
        Asn1Utf8String { content }
    }
    pub fn with_str(s: &str) -> Self {
        Asn1Utf8String::new(s.to_string())
    }
}

impl Display for Asn1Utf8String {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.content)
    }
}

impl From<Asn1Utf8String> for String {
    fn from(value: Asn1Utf8String) -> Self {
        value.content
    }
}