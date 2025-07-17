use crate::Result;
use crate::asn1::{Asn1Object, Asn1ObjectIdentifier, Asn1Sequence, Asn1TaggedObject};
use crate::asn1::try_from_tagged::{TryFromTagged, TryIntoTagged};

/// Represents an ASN.1 AlgorithmIdentifier structure.
///
/// ```text
/// AlgorithmIdentifier ::= SEQUENCE {
///     algorithm OBJECT IDENTIFIER,
///     parameters ANY DEFINED BY algorithm OPTIONAL
/// }
/// ```
pub struct AlgorithmIdentifier {
    algorithm: Asn1ObjectIdentifier,
    parameters: Option<Asn1Object>,
}

impl AlgorithmIdentifier {
    pub fn new(algorithm: Asn1ObjectIdentifier, parameters: Option<Asn1Object>) -> Self {
        AlgorithmIdentifier { algorithm, parameters }
    }
    pub fn with_sequence(sequence: Asn1Sequence) -> Result<Self> {
        if sequence.len() < 1 || sequence.len() > 2 {
            return Err(crate::BcError::with_invalid_argument(format!("Bad sequence size: {}", sequence.len())));
        }

        let mut iter = sequence.into_iter();
        let algorithm = iter.next().unwrap().try_into()?;
        let parameters = if let Some(param) = iter.next() { Some(param) } else { None };
        Ok(AlgorithmIdentifier { algorithm, parameters })
    }
    pub fn get_algorithm(&self) -> &Asn1ObjectIdentifier {
        &self.algorithm
    }
    pub fn get_parameters(&self) -> Option<&Asn1Object> {
        self.parameters.as_ref()
    }
}
impl From<AlgorithmIdentifier> for Asn1Object  {
    fn from(value: AlgorithmIdentifier) -> Self {
        let mut sequence = vec![value.algorithm.into()];
        if let Some(params) = value.parameters {
            sequence.push(params);
        }
        Asn1Sequence::new(sequence).into()
    }
}
impl TryFrom<Asn1Object> for AlgorithmIdentifier {
    type Error = crate::BcError;

    fn try_from(asn1_object: Asn1Object) -> Result<Self> {
        if let Asn1Object::Sequence(sequence) = asn1_object {
            Self::with_sequence(sequence)
        } else {
            Err(crate::BcError::with_invalid_cast("Expected a sequence for AlgorithmIdentifier"))
        }
    }
}
impl TryFromTagged for AlgorithmIdentifier {
    fn try_from_tagged(tagged: Asn1TaggedObject, declared_explicit: bool) -> Result<Self>
    where
        Self: Sized
    {
        let sequence: Asn1Sequence = tagged.try_into_tagged(declared_explicit)?;
        Self::with_sequence(sequence)
    }
}
// TODO
