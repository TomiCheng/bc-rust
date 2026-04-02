//! PEM writer.
//!
//! Port of `PemWriter.cs` from bc-csharp.

use std::io::Write;
use crate::error::BcResult;
use crate::util::encoders::base64::{Base64Alphabet, Base64Encoder};
use super::pem_object_generator::PemObjectGenerator;

/// Line length for Base64-encoded PEM content, as per RFC 1421.
const LINE_LENGTH: usize = 64;

/// Writes PEM-formatted data to a [`Write`] output.
///
/// Equivalent to bc-csharp's `PemWriter`.
///
/// # Examples
///
/// ```
/// use bc_rust::util::io::pem::pem_object::PemObject;
/// use bc_rust::util::io::pem::pem_writer::PemWriter;
///
/// let obj = PemObject::new("CERTIFICATE", vec![0x01, 0x02, 0x03]);
/// let mut out = Vec::new();
/// let mut writer = PemWriter::new(&mut out);
/// writer.write_object(&obj).unwrap();
///
/// let pem = String::from_utf8(out).unwrap();
/// assert!(pem.starts_with("-----BEGIN CERTIFICATE-----\n"));
/// assert!(pem.ends_with("-----END CERTIFICATE-----\n"));
/// ```
pub struct PemWriter<W: Write> {
    writer: W,
}

impl<W: Write> PemWriter<W> {
    /// Creates a new `PemWriter` wrapping `writer`.
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    /// Returns the number of bytes required to encode `obj` as PEM.
    ///
    /// Equivalent to bc-csharp's `PemWriter.GetOutputSize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use bc_rust::util::io::pem::pem_object::PemObject;
    /// use bc_rust::util::io::pem::pem_writer::PemWriter;
    ///
    /// let obj = PemObject::new("CERTIFICATE", vec![0x01, 0x02, 0x03]);
    /// let mut out = Vec::new();
    /// let mut writer = PemWriter::new(&mut out);
    /// let size = writer.get_output_size(&obj);
    /// writer.write_object(&obj).unwrap();
    /// assert_eq!(out.len(), size);
    /// ```
    pub fn get_output_size(&self, obj: &super::pem_object::PemObject) -> usize {
        // BEGIN + END lines: "-----BEGIN TYPE-----\n" + "-----END TYPE-----\n"
        let mut size = 2 * (obj.pem_type().len() + 10 + 1) + 10;

        // Headers + empty separator line
        if !obj.headers().is_empty() {
            for h in obj.headers() {
                size += h.name().len() + 2 + h.value().len() + 1;
            }
            size += 1; // empty line
        }

        // Base64 content + newlines
        let data_len = obj.content().len().div_ceil(3) * 4;
        size += data_len + data_len.div_ceil(LINE_LENGTH);

        size
    }

    /// Writes a PEM object to the output.
    ///
    /// # Errors
    ///
    /// - Returns [`crate::error::BcError::PemError`] if generation fails.
    /// - Returns [`crate::error::BcError::IoError`] if writing fails.
    pub fn write_object(&mut self, obj: &impl PemObjectGenerator) -> BcResult<()> {
        let obj = obj.generate()?;

        writeln!(self.writer, "-----BEGIN {}-----", obj.pem_type())?;

        if !obj.headers().is_empty() {
            for header in obj.headers() {
                writeln!(self.writer, "{}: {}", header.name(), header.value())?;
            }
            writeln!(self.writer)?;
        }

        let encoder = Base64Encoder::new(Base64Alphabet::Standard);
        let encoded = encoder.to_base64_string(obj.content());
        for chunk in encoded.as_bytes().chunks(LINE_LENGTH) {
            self.writer.write_all(chunk)?;
            writeln!(self.writer)?;
        }

        writeln!(self.writer, "-----END {}-----", obj.pem_type())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::io::pem::pem_header::PemHeader;
    use crate::util::io::pem::pem_object::PemObject;

    fn write_to_string(obj: &PemObject) -> String {
        let mut out = Vec::new();
        PemWriter::new(&mut out).write_object(obj).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn test_write_simple() {
        let obj = PemObject::new("CERTIFICATE", vec![0x01, 0x02, 0x03]);
        let pem = write_to_string(&obj);
        assert!(pem.starts_with("-----BEGIN CERTIFICATE-----\n"));
        assert!(pem.ends_with("-----END CERTIFICATE-----\n"));
    }

    #[test]
    fn test_write_with_headers() {
        let headers = vec![PemHeader::new("Proc-Type", "4,ENCRYPTED")];
        let obj = PemObject::with_headers("RSA PRIVATE KEY", headers, vec![0xAB, 0xCD]);
        let pem = write_to_string(&obj);
        assert!(pem.contains("Proc-Type: 4,ENCRYPTED\n"));
        assert!(pem.contains("-----BEGIN RSA PRIVATE KEY-----\n"));
        assert!(pem.contains("-----END RSA PRIVATE KEY-----\n"));
    }

    #[test]
    fn test_write_long_content_wraps_at_64() {
        let content = vec![0xABu8; 60]; // produces 80 Base64 chars
        let obj = PemObject::new("DATA", content);
        let pem = write_to_string(&obj);
        for line in pem.lines() {
            if line.starts_with("-----") { continue; }
            assert!(line.len() <= LINE_LENGTH, "line too long: {}", line.len());
        }
    }

    fn length_test(pem_type: &str, headers: Vec<PemHeader>, content: Vec<u8>) {
        let obj = PemObject::with_headers(pem_type, headers, content);
        let mut out = Vec::new();
        let mut writer = PemWriter::new(&mut out);
        let size = writer.get_output_size(&obj);
        writer.write_object(&obj).unwrap();
        assert_eq!(out.len(), size, "size mismatch for content len {}", obj.content().len());
    }

    #[test]
    fn test_pem_length_various_sizes() {
        for i in 1..60 {
            length_test("CERTIFICATE", vec![], vec![0u8; i]);
        }
        for i in [100, 101, 102, 103, 1000, 1001, 1002, 1003] {
            length_test("CERTIFICATE", vec![], vec![0u8; i]);
        }
    }

    #[test]
    fn test_pem_length_with_headers() {
        let headers = vec![
            PemHeader::new("Proc-Type", "4,ENCRYPTED"),
            PemHeader::new("DEK-Info", "DES3,0001020304050607"),
        ];
        length_test("RSA PRIVATE KEY", headers, vec![0u8; 103]);
    }

    #[test]
    fn test_write_empty_content() {
        let obj = PemObject::new("CERTIFICATE", vec![]);
        let pem = write_to_string(&obj);
        assert_eq!(pem, "-----BEGIN CERTIFICATE-----\n-----END CERTIFICATE-----\n");
    }
}
