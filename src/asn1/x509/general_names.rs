use crate::asn1::{Asn1Object, Asn1Sequence};
use crate::asn1::x509::GeneralName;
use crate::BcError;
use crate::Result;
pub struct GeneralNames {
    names: Vec<GeneralName>
}

impl GeneralNames {
    fn new(names: Vec<GeneralName>) -> Self {
        GeneralNames { names }
    }
    pub(crate) fn from_asn1_object(asn1_object: Asn1Object) -> Result<Self> {
        if let Asn1Object::Sequence(sequence) = asn1_object {
            Self::from_sequence(sequence)
        } else {
            Err(BcError::with_invalid_argument("Expected a sequence for GeneralNames"))
        }
    }

    fn from_sequence(sequence: Asn1Sequence) -> Result<Self> {
        let mut names = Vec::with_capacity(sequence.len());
        for element in sequence.into_iter() {
            let name = GeneralName::from_asn1_object(element)?;
            names.push(name);
        }
        Ok(Self::new(names))
    }
    
    pub fn into_iter(self) -> std::vec::IntoIter<GeneralName> {
        self.names.into_iter()
    }
}