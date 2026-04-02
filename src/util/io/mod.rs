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
//! | `BufferedFilterStream.cs` | — | Not ported — replaced by [`std::io::BufReader`] / [`std::io::BufWriter`] |
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
