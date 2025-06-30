use crate::asn1::{Asn1Convertible, Asn1Object, Asn1ObjectIdentifier};
use crate::Result;

pub struct AlgorithmIdentifier {
    algorithm: Asn1ObjectIdentifier,
    parameters: Option<Asn1Object>
}

impl AlgorithmIdentifier {
    pub(crate) fn from_asn1_object(asn1_object: &Asn1Object) -> Result<Self> {
        if let Some(sequence) = asn1_object.as_sequence() {
            if sequence.len() < 1 || sequence.len() > 2 {
                return Err(crate::BcError::with_invalid_argument(format!("Bad sequence size: {}", sequence.len())));
            }

            let algorithm = Asn1ObjectIdentifier::from_asn1_object(&sequence[0])?;
            let parameters = if sequence.len() == 2 {
                Some(sequence[1].clone())
            } else {
                None
            };
            Ok(AlgorithmIdentifier { algorithm, parameters })
        } else {
            Err(crate::BcError::with_invalid_cast("Expected a sequence for AlgorithmIdentifier"))
        }
    }
}

impl Asn1Convertible for AlgorithmIdentifier {
    fn to_asn1_object(&self) -> Result<Asn1Object> {
        todo!()
    }
}
// TODO