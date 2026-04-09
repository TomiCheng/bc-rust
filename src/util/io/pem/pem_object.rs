//! PEM object containing type, headers and Base64-encoded content.
//!
//! Port of `PemObject.cs` from bc-csharp.

use super::pem_header::PemHeader;
use super::pem_object_generator::PemObjectGenerator;
use crate::error::BcResult;

/// A PEM object with a type, optional headers, and binary content.
///
/// Represents a single PEM block, e.g.:
///
/// ```text
/// -----BEGIN CERTIFICATE-----
/// Proc-Type: 4,ENCRYPTED
///
/// Base64EncodedContent...
/// -----END CERTIFICATE-----
/// ```
///
/// # Examples
///
/// ```
/// use bc_rust::util::io::pem::pem_object::PemObject;
///
/// let obj = PemObject::new("CERTIFICATE", vec![0x01, 0x02, 0x03]);
/// assert_eq!(obj.pem_type(), "CERTIFICATE");
/// assert!(obj.headers().is_empty());
/// assert_eq!(obj.content(), &[0x01, 0x02, 0x03]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PemObject {
    pem_type: String,
    headers: Vec<PemHeader>,
    content: Vec<u8>,
}

impl PemObject {
    /// Creates a new `PemObject` with the given type and content, and no headers.
    pub fn new(pem_type: impl Into<String>, content: Vec<u8>) -> Self {
        Self {
            pem_type: pem_type.into(),
            headers: Vec::new(),
            content,
        }
    }

    /// Creates a new `PemObject` with the given type, headers, and content.
    pub fn with_headers(
        pem_type: impl Into<String>,
        headers: Vec<PemHeader>,
        content: Vec<u8>,
    ) -> Self {
        Self {
            pem_type: pem_type.into(),
            headers,
            content,
        }
    }

    /// Returns the PEM type (e.g. `"CERTIFICATE"`, `"RSA PRIVATE KEY"`).
    pub fn pem_type(&self) -> &str {
        &self.pem_type
    }

    /// Returns the PEM headers.
    pub fn headers(&self) -> &[PemHeader] {
        &self.headers
    }

    /// Returns the binary content.
    pub fn content(&self) -> &[u8] {
        &self.content
    }
}

impl PemObjectGenerator for PemObject {
    fn generate(&self) -> BcResult<PemObject> {
        Ok(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let obj = PemObject::new("CERTIFICATE", vec![0x01, 0x02, 0x03]);
        assert_eq!(obj.pem_type(), "CERTIFICATE");
        assert!(obj.headers().is_empty());
        assert_eq!(obj.content(), &[0x01, 0x02, 0x03]);
    }

    #[test]
    fn test_with_headers() {
        let headers = vec![PemHeader::new("Proc-Type", "4,ENCRYPTED")];
        let obj = PemObject::with_headers("RSA PRIVATE KEY", headers.clone(), vec![0xAB]);
        assert_eq!(obj.pem_type(), "RSA PRIVATE KEY");
        assert_eq!(obj.headers(), headers.as_slice());
        assert_eq!(obj.content(), &[0xAB]);
    }

    #[test]
    fn test_clone() {
        let obj = PemObject::new("CERTIFICATE", vec![0x01]);
        assert_eq!(obj.clone(), obj);
    }
}
