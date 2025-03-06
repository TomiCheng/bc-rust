use crate::asn1::{Asn1Sequence, DerBitString};

use super::{AlgorithmIdentifier, TbsCertificateStructure};

/// an X509Certificate structure.
/// ```text
/// Certificate ::= SEQUENCE {
///   tbsCertificate      TBSCertificate,
///   signatureAlgorithm  AlgorithmIdentifier,
///   signatureValue      BitString
/// }
/// ```
pub struct X509CertificateStructure {
    tbs_certificate: TbsCertificateStructure,
    sign_alg_id: AlgorithmIdentifier,
    sign: DerBitString,
}

impl X509CertificateStructure {
    pub fn new(
        tbs_certificate: TbsCertificateStructure,
        sign_alg_id: AlgorithmIdentifier,
        sign: DerBitString,
    ) -> Self {
        X509CertificateStructure {
            tbs_certificate,
            sign_alg_id,
            sign,
        }
    }

    pub fn get_algorithm_identifier(&self) -> &AlgorithmIdentifier {
        &self.sign_alg_id
    }

    pub fn get_signature(&self) -> &DerBitString {
        &self.sign
    }
}

impl TryFrom<&Asn1Sequence> for X509CertificateStructure {
    type Error = crate::BcError;

    fn try_from(seq: &Asn1Sequence) -> Result<Self, Self::Error> {
        let count = seq.len();
        if count != 3 {

        }
        todo!()
    }
}


