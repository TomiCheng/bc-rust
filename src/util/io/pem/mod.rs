//! PEM encoding and decoding utilities.
//!
//! Port of `util/io/pem/` from bc-csharp.
//!
//! PEM (Privacy-Enhanced Mail) is a Base64-encoded format with header/footer
//! markers such as `-----BEGIN CERTIFICATE-----` and `-----END CERTIFICATE-----`.
//!
//! ## bc-csharp mapping
//!
//! | bc-csharp | bc-rust | Notes |
//! |-----------|---------|-------|
//! | `PemGenerationException.cs` | — | Not ported — use [`crate::error::BcError`] instead |
//! | `PemHeader.cs` | [`pem_header::PemHeader`] | PEM header name-value pair |
//! | `PemObject.cs` | [`pem_object::PemObject`] | PEM object with type, headers and content |
//! | `PemObjectGenerator.cs` | [`pem_object_generator::PemObjectGenerator`] | Trait for generating PEM objects |
//! | `PemObjectParser.cs` | [`pem_object_parser::PemObjectParser`] | Trait for parsing PEM objects |
//! | `PemReader.cs` | [`pem_reader::PemReader`] | Reads PEM-formatted data |
//! | `PemWriter.cs` | [`pem_writer::PemWriter`] | Writes PEM-formatted data |

pub mod pem_header;
pub mod pem_object;
pub mod pem_object_generator;
pub mod pem_object_parser;
pub mod pem_reader;
pub mod pem_writer;
