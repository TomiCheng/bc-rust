use crate::asn1::{Asn1BitString, Asn1Object, Asn1Sequence};
use crate::asn1::x509::AlgorithmIdentifier;
use crate::Result;
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
    pub(crate) fn from_asn1_object(asn1_object: Asn1Object) -> Result<Self> {
        if let Ok(sequence) = asn1_object.try_into() {
            return SubjectPublicKeyInfo::from_sequence(sequence);
        }
        todo!()
    }
    fn from_sequence(sequence: Asn1Sequence) -> Result<Self> {
        if sequence.len() != 2 {
            return Err(crate::BcError::with_invalid_format(format!("bad sequence size: {}", sequence.len())));
        }
        let mut iter = sequence.into_iter();
        let algorithm = AlgorithmIdentifier::from_asn1_object(iter.next().unwrap())?;
        let subject_public_key =  iter.next().unwrap().try_into()?;
        
        Ok(SubjectPublicKeyInfo::new(algorithm, subject_public_key))
    }
}
