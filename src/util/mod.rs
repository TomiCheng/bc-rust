//! Utility modules for the bc-rust library.
//!
//! ## bc-csharp mapping
//!
//! | bc-csharp | bc-rust | Notes |
//! |-----------|---------|-------|
//! | `util/date/DateTimeUtilities.cs` | [`date`] | Unix timestamp utilities |
//! | `util/io/` | [`io`] | I/O utility types and functions |
//! | `util/encoders/` | [`encoders`] | Hex / Base64 encoding and decoding |
//! | `util/Integers.cs` | [`integers`] | Bit manipulation for `u32` |
//! | `util/Longs.cs` | [`longs`] | Bit manipulation for `u64` |
//! | `util/Shorts.cs` | [`shorts`] | Bit manipulation for `u16` |
//! | `util/bzip2/` | [`bzip2`] | BZip2 compression and decompression |
//! | `util/Arrays.cs` | [`arrays`] | Array utilities (zeroing, constant-time compare, copy, concat) |
//! | `util/net/IPAddress.cs` | — | Not ported — covered by Rust `std::net` |

pub mod arrays;
pub mod bzip2;
pub mod date;
pub mod encoders;
pub mod integers;
pub mod io;
pub mod longs;
pub mod shorts;
