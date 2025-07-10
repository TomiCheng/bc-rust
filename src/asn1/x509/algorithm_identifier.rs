use crate::Result;
use crate::asn1::{Asn1Object, Asn1ObjectIdentifier};

pub struct AlgorithmIdentifier {
    algorithm: Asn1ObjectIdentifier,
    parameters: Option<Asn1Object>,
}

impl AlgorithmIdentifier {
    pub fn new(algorithm: Asn1ObjectIdentifier, parameters: Option<Asn1Object>) -> Self {
        AlgorithmIdentifier { algorithm, parameters }
    }

    pub fn algorithm(&self) -> &Asn1ObjectIdentifier {
        &self.algorithm
    }

    pub fn parameters(&self) -> Option<&Asn1Object> {
        self.parameters.as_ref()
    }
}
impl TryFrom<Asn1Object> for AlgorithmIdentifier {
    type Error = crate::BcError;

    fn try_from(asn1_object: Asn1Object) -> Result<Self> {
        if let Asn1Object::Sequence(sequence) = asn1_object.into() {
            let len = sequence.len();
            if len < 1 || len > 2 {
                return Err(crate::BcError::with_invalid_argument(format!("Bad sequence size: {}", sequence.len())));
            }

            let mut iter = sequence.into_iter();
            let algorithm = iter.next().unwrap().try_into()?;
            let parameters = if len == 2 { Some(iter.next().unwrap()) } else { None };
            Ok(AlgorithmIdentifier { algorithm, parameters })
        } else {
            Err(crate::BcError::with_invalid_cast("Expected a sequence for AlgorithmIdentifier"))
        }
    }
}
// TODO
