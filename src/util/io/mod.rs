//! I/O utility types and functions.
//!
//! Port of `util/io/` from bc-csharp.
//!
//! ## bc-csharp mapping
//!
//! | bc-csharp | bc-rust | Notes |
//! |-----------|---------|-------|
//! | `BaseInputStream.cs` | — | Not ported — replaced by [`std::io::Read`] trait |
//! | `BaseOutputStream.cs` | — | Not ported — replaced by [`std::io::Write`] trait |
//! | `BinaryReaders.cs` | — | Not ported — covered by `u32::from_be_bytes` / `from_le_bytes` |
//! | `BinaryWriters.cs` | — | Not ported — covered by `u32::to_be_bytes` / `to_le_bytes` |
//! | `BufferedFilterStream.cs` | — | Not ported — replaced by [`std::io::BufReader`] / [`std::io::BufWriter`] |
//! | `FilterStream.cs` | — | Not ported — Rust uses composition over inheritance |
//! | `LimitedBuffer.cs` | — | Not ported — [`Vec<u8>`] implements [`std::io::Write`] directly |
//! | `StreamOverflowException.cs` | — | Not ported — use [`crate::error::BcError`] instead |
//! | `LimitedInputStream.cs` | [`limited_reader`] | Byte-limited [`std::io::Read`] wrapper |
//! | `PushbackStream.cs` | [`pushback_reader`] | Single-byte pushback [`std::io::Read`] wrapper |
//! | `Streams.cs` | [`streams`] | Stream utility functions |
//! | `TeeInputStream.cs` | [`tee_reader`] | [`std::io::Read`] wrapper that copies to a secondary [`std::io::Write`] |
//! | `TeeOutputStream.cs` | [`tee_writer`] | [`std::io::Write`] wrapper that copies to a secondary [`std::io::Write`] |

pub mod limited_reader;
pub mod pushback_reader;
pub mod streams;
pub mod tee_reader;
pub mod tee_writer;
