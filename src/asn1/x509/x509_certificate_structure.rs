use crate::asn1::x509::{AlgorithmIdentifier, TbsCertificateStructure};
use crate::asn1::{Asn1BitString, Asn1Convertible, Asn1Object, Asn1Sequence, Asn1TaggedObject};
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
    pub fn with_bytes(buffer: &[u8]) -> Result<Self> {
        let asn1_object = Asn1Object::with_bytes(buffer)?;
        let sequence: Asn1Sequence = asn1_object.try_into()?;
        if sequence.len() != 3 {
            return Err(BcError::with_invalid_argument(format!("Bad sequence size: {}", sequence.len())));
        }
        let mut array: Vec<Asn1Object> = sequence.into();
        Ok(X509CertificateStructure {
            tbs_certificate: TbsCertificateStructure::from_asn1_object(array.remove(0))?,
            algorithm: AlgorithmIdentifier::from_asn1_object(array.remove(0))?,
            signature: Asn1BitString::from_asn1_object(array.remove(0))?,
        })
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
    use crate::asn1::x509::{x509_extensions, ExtendedKeyUsage, GeneralNames, KeyPurposeId, KeyUsage, SubjectPublicKeyInfo, X509CertificateStructure};
    use base64::prelude::*;
    use crate::asn1::{Asn1Convertible, Asn1Object};

    const SUBECTS: [&'static str; 7] = [
        "C=AU,ST=Victoria,L=South Melbourne,O=Connect 4 Pty Ltd,OU=Webserver Team,CN=www2.connect4.com.au,E=webmaster@connect4.com.au",
        "C=AU,ST=Victoria,L=South Melbourne,O=Connect 4 Pty Ltd,OU=Certificate Authority,CN=Connect 4 CA,E=webmaster@connect4.com.au",
        "C=AU,ST=QLD,CN=SSLeay/rsa test cert",
        "C=US,O=National Aeronautics and Space Administration,SERIALNUMBER=16+CN=Steve Schoch",
        "E=cooke@issl.atl.hp.com,C=US,OU=Hewlett Packard Company (ISSL),CN=Paul A. Cooke",
        "O=Sun Microsystems Inc,CN=store.sun.com",
        "unstructuredAddress=192.168.1.33,unstructuredName=pixfirewall.ciscopix.com,CN=pixfirewall.ciscopix.com",
    ];
    #[test]
    fn test_parse_x509_certificate_01() {
        const CERTIFICATE_BASE64: &str = concat!(
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
        let certificate_buffer = BASE64_STANDARD.decode(CERTIFICATE_BASE64).unwrap();
        check_certificate(1, &certificate_buffer);
    }

    fn check_certificate(id: usize, certificate_buffer: &[u8]) {
        let obj = X509CertificateStructure::with_bytes(certificate_buffer).unwrap();
        let tbs_certificate = obj.tbs_certificate();
        assert_eq!(SUBECTS[id - 1], &tbs_certificate.subject().to_string());

        if tbs_certificate.version() >= 3 {
            if let Some(extensions) = tbs_certificate.extensions() {
                for oid in extensions.iter_ordering() {
                    let extension = extensions.get_extension(oid).unwrap();
                    let extension_object = Asn1Object::with_bytes(extension.get_value().get_octets()).unwrap();
                    
                    if oid == &(*x509_extensions::SUBJECT_KEY_IDENTIFIER) {
                        SubjectPublicKeyInfo::from_asn1_object(extension_object).unwrap();
                    } else if oid == &(*x509_extensions::KEY_USAGE) {
                        KeyUsage::from_asn1_object(extension_object).unwrap();
                    } else if oid == &(*x509_extensions::EXTENDED_KEY_USAGE) {
                        let extended_key_usage = ExtendedKeyUsage::from_asn1_object(extension_object).unwrap();
                        for usage in extended_key_usage.into_iter() {
                            KeyPurposeId::new(usage);
                        }
                    } else if oid == &(*x509_extensions::SUBJECT_ALTERNATIVE_NAME) {
                        let general_names = GeneralNames::from_asn1_object(extension_object).unwrap();
                        for name in general_names.into_iter() {
                            name.to_asn1_object().unwrap();
                        }
                    }
                
                        
                }
            }
        }
    }
}
