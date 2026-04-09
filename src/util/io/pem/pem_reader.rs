//! PEM reader.
//!
//! Port of `PemReader.cs` from bc-csharp.
//!
//! Unlike bc-csharp which uses character-by-character parsing with a pushback
//! stack, this implementation uses [`std::io::BufRead`] for line-oriented
//! reading, which is simpler and idiomatic in Rust.

use super::pem_header::PemHeader;
use super::pem_object::PemObject;
use crate::error::{BcError, BcResult};
use crate::util::encoders::base64::{Base64Alphabet, Base64Encoder};
use std::io::BufRead;

/// Reads PEM-formatted data from a [`BufRead`] input.
///
/// Equivalent to bc-csharp's `PemReader`.
///
/// # Examples
///
/// ```
/// use bc_rust::util::io::pem::pem_reader::PemReader;
///
/// let pem = b"-----BEGIN CERTIFICATE-----\nAQID\n-----END CERTIFICATE-----\n";
/// let mut reader = PemReader::new(pem.as_ref());
/// let obj = reader.read_pem_object().unwrap().unwrap();
/// assert_eq!(obj.pem_type(), "CERTIFICATE");
/// assert_eq!(obj.content(), &[0x01, 0x02, 0x03]);
/// ```
pub struct PemReader<R: BufRead> {
    reader: R,
}

impl<R: BufRead> PemReader<R> {
    /// Creates a new `PemReader` wrapping `reader`.
    pub fn new(reader: R) -> Self {
        Self { reader }
    }

    /// Reads the next PEM object from the input.
    ///
    /// Returns `Ok(None)` at end of input.
    ///
    /// # Errors
    ///
    /// Returns [`BcError::PemError`] if the PEM data is malformed.
    /// Returns [`BcError::IoError`] if reading fails.
    pub fn read_pem_object(&mut self) -> BcResult<Option<PemObject>> {
        let pem_type = match self.find_begin()? {
            Some(t) => t,
            None => return Ok(None),
        };

        let mut headers = Vec::new();
        let mut base64_data = String::new();
        let mut in_headers = true;
        let mut line = String::new();

        loop {
            line.clear();
            if self.reader.read_line(&mut line)? == 0 {
                return Err(BcError::PemError {
                    msg: "unexpected end of input before END marker".to_string(),
                });
            }

            let trimmed = line.trim();

            // END marker
            if trimmed.starts_with("-----END ") && trimmed.ends_with("-----") {
                let end_type = &trimmed[9..trimmed.len() - 5];
                if end_type != pem_type {
                    return Err(BcError::PemError {
                        msg: format!("expected END {}, got END {}", pem_type, end_type),
                    });
                }
                break;
            }

            // Empty line separates headers from content
            if trimmed.is_empty() {
                in_headers = false;
                continue;
            }

            // Header line (key: value)
            if in_headers {
                if let Some(pos) = trimmed.find(':') {
                    let key = trimmed[..pos].trim();
                    let val = trimmed[pos + 1..].trim();
                    headers.push(PemHeader::new(key, val));
                    continue;
                }
                in_headers = false; // no colon — it's content
            }

            base64_data.push_str(trimmed);
        }

        let encoder = Base64Encoder::new(Base64Alphabet::Standard);
        let content = encoder.decode_str(&base64_data)?;

        Ok(Some(PemObject::with_headers(pem_type, headers, content)))
    }

    /// Scans forward for a `-----BEGIN TYPE-----` marker.
    ///
    /// Returns `Ok(None)` if EOF is reached without finding one.
    fn find_begin(&mut self) -> BcResult<Option<String>> {
        let mut line = String::new();
        loop {
            line.clear();
            if self.reader.read_line(&mut line)? == 0 {
                return Ok(None);
            }
            let trimmed = line.trim();
            if trimmed.starts_with("-----BEGIN ") && trimmed.ends_with("-----") {
                let pem_type = trimmed[11..trimmed.len() - 5].trim().to_string();
                if pem_type.is_empty() {
                    return Err(BcError::PemError {
                        msg: "empty PEM type in BEGIN marker".to_string(),
                    });
                }
                return Ok(Some(pem_type));
            } else if trimmed.starts_with("-----BEGIN") {
                return Err(BcError::PemError {
                    msg: "malformed PEM BEGIN marker".to_string(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_read_simple() {
        let pem = b"-----BEGIN CERTIFICATE-----\nAQID\n-----END CERTIFICATE-----\n";
        let mut reader = PemReader::new(pem.as_ref());
        let obj = reader.read_pem_object().unwrap().unwrap();
        assert_eq!(obj.pem_type(), "CERTIFICATE");
        assert_eq!(obj.content(), &[0x01, 0x02, 0x03]);
        assert!(obj.headers().is_empty());
    }

    #[test]
    fn test_read_with_headers() {
        let pem = "-----BEGIN RSA PRIVATE KEY-----\nProc-Type: 4,ENCRYPTED\nDEK-Info: AES-128-CBC,ABC\n\nAQID\n-----END RSA PRIVATE KEY-----\n";
        let mut reader = PemReader::new(pem.as_bytes());
        let obj = reader.read_pem_object().unwrap().unwrap();
        assert_eq!(obj.pem_type(), "RSA PRIVATE KEY");
        assert_eq!(obj.headers().len(), 2);
        assert_eq!(obj.headers()[0].name(), "Proc-Type");
        assert_eq!(obj.headers()[0].value(), "4,ENCRYPTED");
    }

    #[test]
    fn test_read_eof() {
        let pem = b"no pem here";
        let mut reader = PemReader::new(pem.as_ref());
        assert!(reader.read_pem_object().unwrap().is_none());
    }

    #[test]
    fn test_read_ignores_leading_text() {
        let pem = "some text\n-----BEGIN CERTIFICATE-----\nAQID\n-----END CERTIFICATE-----\n";
        let mut reader = PemReader::new(pem.as_bytes());
        let obj = reader.read_pem_object().unwrap().unwrap();
        assert_eq!(obj.pem_type(), "CERTIFICATE");
    }

    #[test]
    fn test_roundtrip() {
        use crate::util::io::pem::pem_writer::PemWriter;

        let original = PemObject::new("CERTIFICATE", vec![0x01, 0x02, 0x03, 0x04, 0x05]);
        let mut buf = Vec::new();
        PemWriter::new(&mut buf).write_object(&original).unwrap();

        let mut reader = PemReader::new(Cursor::new(buf));
        let parsed = reader.read_pem_object().unwrap().unwrap();
        assert_eq!(parsed.pem_type(), original.pem_type());
        assert_eq!(parsed.content(), original.content());
    }

    #[test]
    fn test_malformed_begin() {
        // "-----BEGIN \n" — BEGIN with no type, port of bc-csharp TestMalformed
        let pem = "-----BEGIN \n";
        let mut reader = PemReader::new(pem.as_bytes());
        assert!(reader.read_pem_object().is_err());
    }

    #[test]
    fn test_wrong_end_type() {
        let pem = "-----BEGIN CERTIFICATE-----\nAQID\n-----END OTHER-----\n";
        let mut reader = PemReader::new(pem.as_bytes());
        assert!(reader.read_pem_object().is_err());
    }
}
