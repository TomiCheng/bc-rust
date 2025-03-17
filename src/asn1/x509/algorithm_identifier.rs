use crate::asn1;

pub struct AlgorithmIdentifier {
    algorithm: asn1::DerObjectIdentifier,
    parameters: Option<Box<dyn asn1::Asn1Convertiable>>,
}

impl AlgorithmIdentifier {
    pub fn new(
        algorithm: asn1::DerObjectIdentifier,
        parameters: Option<Box<dyn asn1::Asn1Convertiable>>,
    ) -> Self {
        AlgorithmIdentifier {
            algorithm,
            parameters,
        }
    }
}