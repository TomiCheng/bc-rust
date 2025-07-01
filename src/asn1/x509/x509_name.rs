use crate::asn1::Asn1Object;
use crate::Result;

/// ```text
/// RDNSequence ::= SEQUENCE OF RelativeDistinguishedName
/// 
/// RelativeDistinguishedName ::= SET SIZE (1..MAX) OF AttributeTypeAndValue
/// 
/// AttributeTypeAndValue ::= SEQUENCE {
///    type AttributeType,
///    value AttributeValue 
/// }
/// ```
pub struct X509Name {
    contents: Vec<String>,
}

impl X509Name {
    pub(crate) fn from_asn1_object(asn1_object: &Asn1Object) -> Result<Self> {
        
        todo!()
        
    }
}
// TODO