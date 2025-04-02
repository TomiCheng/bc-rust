use std::any;
use std::sync;

use crate::asn1;
use crate::BcError;
use crate::Result;

#[derive(Debug, Clone)]
pub struct AlgorithmIdentifier {
    algorithm: asn1::DerObjectIdentifier,
    parameters: Option<sync::Arc<dyn asn1::Asn1Encodable>>,
}

impl AlgorithmIdentifier {
    fn new(
        algorithm: asn1::DerObjectIdentifier,
        parameters: Option<sync::Arc<dyn asn1::Asn1Encodable>>,
    ) -> Self {
        AlgorithmIdentifier {
            algorithm,
            parameters,
        }
    }

    pub fn with_algroithm(algorithm: asn1::DerObjectIdentifier) -> Self {
        AlgorithmIdentifier::new(algorithm, None)
    }

    pub fn with_asn1_sequence(sequence: asn1::Asn1Sequence) -> Result<Self> {
        anyhow::ensure!(
            sequence.len() == 1 || sequence.len() == 2,
            BcError::invalid_argument("Invalid sequence length", "sequence")
        );

        let s1 = sequence[0].clone();
        let s2 = if sequence.len() == 2 {
            Some(sequence[1].clone())
        } else {
            None
        };

        let algorithm = asn1::DerObjectIdentifier::with_asn1_convertiable(s1)?;
        Ok(Self::new(algorithm, s2))
    }

    /// Return the OID in the Algorithm entry of this identifier.
    pub fn algorithm(&self) -> &asn1::DerObjectIdentifier {
        &self.algorithm
    }

    /// Return the parameters structure in the Parameters entry of this identifier.
    pub fn parameters(&self) -> &Option<sync::Arc<dyn asn1::Asn1Encodable>> {
        &self.parameters
    }
}

// trait
impl asn1::Asn1Convertiable for AlgorithmIdentifier {
    fn to_asn1_object(&self) -> sync::Arc<dyn asn1::Asn1Object> {
        if let Some(parameters) = &self.parameters {
            return sync::Arc::new(asn1::DerSequence::with_asn1_encodables(vec![
                sync::Arc::new(self.algorithm.clone()),
                parameters.clone(),
            ]));
        } else {
            return sync::Arc::new(asn1::DerSequence::with_asn1_encodables(vec![
                sync::Arc::new(self.algorithm.clone()),
            ]));
        }
    }

    fn as_any(&self) -> sync::Arc<dyn any::Any> {
        sync::Arc::new(self.clone())
    }
}
