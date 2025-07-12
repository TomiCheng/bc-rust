use crate::asn1::Asn1String;
use std::hash::{Hash};

#[derive(Debug, Clone, Hash, PartialEq)]
pub struct Asn1T61String {
    content: String,
}

impl Asn1String for Asn1T61String {
    fn to_asn1_string(&self) -> crate::Result<String> {
        Ok(self.content.clone())
    }
}
