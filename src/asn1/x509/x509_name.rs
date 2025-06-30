use crate::asn1::Asn1Object;
use crate::Result;

pub struct X509Name {
    contents: Vec<String>,
}

impl X509Name {
    pub(crate) fn from_asn1_object(asn1_object: &Asn1Object) -> Result<Self> {
        todo!()
    }
}
// TODO