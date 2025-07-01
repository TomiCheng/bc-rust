use crate::asn1::{Asn1Object, Asn1ObjectIdentifier, Asn1Sequence, Asn1Set};
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
    ordering: Vec<Asn1ObjectIdentifier>,
    contents: Vec<String>,
}

impl X509Name {
    fn with_sequence(sequence: &Asn1Sequence) -> Result<Self> {
        for element in sequence.get_elements() {
            let rdn_set = Asn1Set::from_asn1_object(element)?;
            for rdn in rdn_set.get_elements() {
                let attribute_type_and_value = Asn1Sequence::from_asn1_object(rdn)?;
                if attribute_type_and_value.len() != 2 {
                    return Err(crate::BcError::with_invalid_format("badly sized AttributeTypeAndValue"));
                }
                
                let type_object = &attribute_type_and_value[0];
                let value_object = &attribute_type_and_value[1];
                
                
            }
        }
        
        todo!();
    }
    pub(crate) fn from_asn1_object(asn1_object: &Asn1Object) -> Result<Self> {
        if let Some(sequence) = asn1_object.as_sequence() {
            return X509Name::with_sequence(sequence);
        }
        //get_instance_choice()
        todo!()
        
    }
}
// TODO