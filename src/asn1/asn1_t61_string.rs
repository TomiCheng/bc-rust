use crate::asn1::Asn1String;

#[derive(Debug, Clone)]
pub struct Asn1T61String {
    content: String,
}

impl Asn1String for Asn1T61String {
    fn to_asn1_string(&self) -> crate::Result<String> {
        todo!()
    }
}