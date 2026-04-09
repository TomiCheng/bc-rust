//! PEM header name-value pair.
//!
//! Port of `PemHeader.cs` from bc-csharp.

use std::fmt;

/// A PEM header name-value pair.
///
/// PEM files may contain optional headers between the `-----BEGIN ...-----`
/// marker and the Base64-encoded content.
///
/// # Examples
///
/// ```
/// use bc_rust::util::io::pem::pem_header::PemHeader;
///
/// let header = PemHeader::new("Proc-Type", "4,ENCRYPTED");
/// assert_eq!(header.name(), "Proc-Type");
/// assert_eq!(header.value(), "4,ENCRYPTED");
/// assert_eq!(header.to_string(), "Proc-Type: 4,ENCRYPTED");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PemHeader {
    name: String,
    value: String,
}

impl PemHeader {
    /// Creates a new `PemHeader` with the given name and value.
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }

    /// Returns the header name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the header value.
    pub fn value(&self) -> &str {
        &self.value
    }
}

impl fmt::Display for PemHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let h = PemHeader::new("Proc-Type", "4,ENCRYPTED");
        assert_eq!(h.name(), "Proc-Type");
        assert_eq!(h.value(), "4,ENCRYPTED");
    }

    #[test]
    fn test_display() {
        let h = PemHeader::new("Proc-Type", "4,ENCRYPTED");
        assert_eq!(h.to_string(), "Proc-Type: 4,ENCRYPTED");
    }

    #[test]
    fn test_equality() {
        let h1 = PemHeader::new("Proc-Type", "4,ENCRYPTED");
        let h2 = PemHeader::new("Proc-Type", "4,ENCRYPTED");
        let h3 = PemHeader::new("DEK-Info", "AES-128-CBC");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_clone() {
        let h1 = PemHeader::new("Proc-Type", "4,ENCRYPTED");
        let h2 = h1.clone();
        assert_eq!(h1, h2);
    }
}
