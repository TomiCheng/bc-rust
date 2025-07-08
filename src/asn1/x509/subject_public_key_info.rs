use crate::asn1::{Asn1BitString, Asn1Object, Asn1Sequence};
use crate::asn1::x509::AlgorithmIdentifier;
use crate::{BcError, Result};
pub struct SubjectPublicKeyInfo {
    algorithm: AlgorithmIdentifier,
    subject_public_key: Asn1BitString,
}

impl SubjectPublicKeyInfo {
    fn new(algorithm: AlgorithmIdentifier, subject_public_key: Asn1BitString) -> Self {
        SubjectPublicKeyInfo {
            algorithm,
            subject_public_key,
        }
    }
    fn from_sequence(sequence: Asn1Sequence) -> Result<Self> {
        if sequence.len() != 2 {
            return Err(crate::BcError::with_invalid_format(format!("bad sequence size: {}", sequence.len())));
        }
        let mut iter = sequence.into_iter();
        let algorithm = iter.next().unwrap().try_into()?;
        let subject_public_key =  iter.next().unwrap().try_into()?;
        
        Ok(SubjectPublicKeyInfo::new(algorithm, subject_public_key))
    }
}
impl TryFrom<Asn1Object> for SubjectPublicKeyInfo {
    type Error = BcError;

    fn try_from(value: Asn1Object) -> Result<Self> {
        if let Asn1Object::Sequence(sequence) = value {
            return SubjectPublicKeyInfo::from_sequence(sequence);
        } 
        todo!()
    }
}
