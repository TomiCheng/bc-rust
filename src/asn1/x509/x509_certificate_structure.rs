use crate::asn1::DerBitStringImpl;

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
    sign: DerBitStringImpl,
}

impl X509CertificateStructure {
    pub fn new(
        tbs_certificate: TbsCertificateStructure,
        sign_alg_id: AlgorithmIdentifier,
        sign: DerBitStringImpl,
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

    pub fn get_signature(&self) -> &DerBitStringImpl {
        &self.sign
    }
}


