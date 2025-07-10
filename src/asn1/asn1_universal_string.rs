use crate::asn1::Asn1String;

#[derive(Debug, Clone, Hash)]
pub struct Asn1UniversalString {
    //content: String,
}

impl Asn1String for Asn1UniversalString {
    fn to_asn1_string(&self) -> crate::Result<String> {
        todo!()
    }
}
