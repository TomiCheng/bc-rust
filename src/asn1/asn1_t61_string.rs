use crate::asn1::Asn1String;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub struct Asn1T61String {
    //content: String,
}

impl Asn1String for Asn1T61String {
    fn to_asn1_string(&self) -> crate::Result<String> {
        todo!()
    }
}
impl Hash for Asn1T61String {
    fn hash<H: Hasher>(&self, state: &mut H) {
        todo!();
    }
}
