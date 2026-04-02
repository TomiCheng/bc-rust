//! Utility modules for the bc-rust library.
//!
//! ## bc-csharp mapping
//!
//! | bc-csharp | bc-rust | Notes |
//! |-----------|---------|-------|
//! | `util/encoders/` | [`encoders`] | Hex / Base64 encoding and decoding |
//! | `util/Integers.cs` | [`integers`] | Bit manipulation for `u32` |
//! | `util/Longs.cs` | [`longs`] | Bit manipulation for `u64` |
//! | `util/Shorts.cs` | [`shorts`] | Bit manipulation for `u16` |
//! | `util/net/IPAddress.cs` | — | Not ported — covered by Rust `std::net` |

pub mod encoders;
pub mod integers;
pub mod longs;
pub mod shorts;
