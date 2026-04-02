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
//! | `Streams.cs` | [`streams`] | Stream utility functions |

pub mod limited_reader;
pub mod streams;
