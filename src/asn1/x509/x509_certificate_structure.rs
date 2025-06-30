use crate::asn1::{Asn1BitString, Asn1Convertible, Asn1Object, Asn1Sequence};
use crate::asn1::x509::{AlgorithmIdentifier, TbsCertificateStructure};
use crate::{BcError, Result};

/// an X509Certificate structure.
/// ```text
/// Certificate ::= Sequence {
///     tbsCertificate TbsCertificate,
///     signatureAlgorithm AlgorithmIdentifier,
///     signature Asn1BitString
/// }
/// ```
pub struct X509CertificateStructure {
    tbs_certificate: TbsCertificateStructure,
    algorithm: AlgorithmIdentifier,
    signature: Asn1BitString,
}

impl X509CertificateStructure {
    /// Creates a new X509CertificateStructure.
    pub fn new(tbs_certificate: TbsCertificateStructure, algorithm: AlgorithmIdentifier, signature: Asn1BitString) -> Self {
        X509CertificateStructure {
            tbs_certificate,
            algorithm,
            signature,
        }
    }
    pub fn from_bytes(buffer: &[u8]) -> Result<Self> {
        let asn1_object = Asn1Object::from_bytes(buffer)?;
        if let Some(sequence) = asn1_object.as_sequence() {
            if sequence.len() != 3 {
                return Err(BcError::with_invalid_argument(format!("Bad sequence size: {}", sequence.len())));
            }
            Ok(X509CertificateStructure {
                tbs_certificate: TbsCertificateStructure::from_asn1_object(&sequence[0])?,
                algorithm: AlgorithmIdentifier::from_asn1_object(&sequence[1])?,
                signature: Asn1BitString::from_asn1_object(&sequence[2])?,
            })
        } else {
            Err(BcError::with_invalid_cast("Expected a sequence for X509CertificateStructure"))
        }
      
    }
    pub fn tbs_certificate(&self) -> &TbsCertificateStructure {
        &self.tbs_certificate
    }
    pub fn signature_algorithm(&self) -> &AlgorithmIdentifier {
        &self.algorithm
    }
    pub fn signature(&self) -> &Asn1BitString {
        &self.signature
    }
}

impl Asn1Convertible for X509CertificateStructure {
    fn to_asn1_object(&self) -> Result<Asn1Object> {
        Ok(Asn1Object::from(Asn1Sequence::new(vec![
            self.tbs_certificate.to_asn1_object()?,
            self.algorithm.to_asn1_object()?,
            self.signature.to_asn1_object()?,
        ])))
    }
}

#[cfg(test)]
mod tests {
    use base64::prelude::*;
    use crate::asn1::x509::X509CertificateStructure;

    #[test]
    fn test_parse_x509_certificate_01() {
        let cert = concat!(
                "MIIDXjCCAsegAwIBAgIBBzANBgkqhkiG9w0BAQQFADCBtzELMAkGA1UEBhMCQVUx",
                "ETAPBgNVBAgTCFZpY3RvcmlhMRgwFgYDVQQHEw9Tb3V0aCBNZWxib3VybmUxGjAY",
                "BgNVBAoTEUNvbm5lY3QgNCBQdHkgTHRkMR4wHAYDVQQLExVDZXJ0aWZpY2F0ZSBB",
                "dXRob3JpdHkxFTATBgNVBAMTDENvbm5lY3QgNCBDQTEoMCYGCSqGSIb3DQEJARYZ",
                "d2VibWFzdGVyQGNvbm5lY3Q0LmNvbS5hdTAeFw0wMDA2MDIwNzU2MjFaFw0wMTA2",
                "MDIwNzU2MjFaMIG4MQswCQYDVQQGEwJBVTERMA8GA1UECBMIVmljdG9yaWExGDAW",
                "BgNVBAcTD1NvdXRoIE1lbGJvdXJuZTEaMBgGA1UEChMRQ29ubmVjdCA0IFB0eSBM",
                "dGQxFzAVBgNVBAsTDldlYnNlcnZlciBUZWFtMR0wGwYDVQQDExR3d3cyLmNvbm5l",
                "Y3Q0LmNvbS5hdTEoMCYGCSqGSIb3DQEJARYZd2VibWFzdGVyQGNvbm5lY3Q0LmNv",
                "bS5hdTCBnzANBgkqhkiG9w0BAQEFAAOBjQAwgYkCgYEArvDxclKAhyv7Q/Wmr2re",
                "Gw4XL9Cnh9e+6VgWy2AWNy/MVeXdlxzd7QAuc1eOWQkGQEiLPy5XQtTY+sBUJ3AO",
                "Rvd2fEVJIcjf29ey7bYua9J/vz5MG2KYo9/WCHIwqD9mmG9g0xLcfwq/s8ZJBswE",
                "7sb85VU+h94PTvsWOsWuKaECAwEAAaN3MHUwJAYDVR0RBB0wG4EZd2VibWFzdGVy",
                "QGNvbm5lY3Q0LmNvbS5hdTA6BglghkgBhvhCAQ0ELRYrbW9kX3NzbCBnZW5lcmF0",
                "ZWQgY3VzdG9tIHNlcnZlciBjZXJ0aWZpY2F0ZTARBglghkgBhvhCAQEEBAMCBkAw",
                "DQYJKoZIhvcNAQEEBQADgYEAotccfKpwSsIxM1Hae8DR7M/Rw8dg/RqOWx45HNVL",
                "iBS4/3N/TO195yeQKbfmzbAA2jbPVvIvGgTxPgO1MP4ZgvgRhasaa0qCJCkWvpM4",
                "yQf33vOiYQbpv4rTwzU8AmRlBG45WdjyNIigGV+oRc61aKCTnLq7zB8N3z1TF/bF",
                "5/8="
            );
        let certificate_buffer = BASE64_STANDARD.decode(cert).unwrap();
        check_certificate(&certificate_buffer);
    }

    fn check_certificate(certificate_buffer: &[u8]) {
        let obj = X509CertificateStructure::from_bytes(certificate_buffer).unwrap();

    }
}