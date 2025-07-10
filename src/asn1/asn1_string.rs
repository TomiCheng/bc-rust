use crate::Result;

pub trait Asn1String {
    fn to_asn1_string(&self) -> Result<String>;
}
