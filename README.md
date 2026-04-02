# bc-rust

A Rust port of [Bouncy Castle C#](https://github.com/bcgit/bc-csharp) — a comprehensive cryptography library.

This is a learning project to explore Rust through practical implementation of cryptographic algorithms.

## Goals

- Port core cryptographic algorithms from bc-csharp to idiomatic Rust
- Learn Rust ownership, traits, generics, and error handling through real-world code

## Status

### Completed

#### `error`
- `BcError` — library-wide error type with variants: `InvalidArgument`, `IoError`, `SystemTimeError`, `InvalidOperation`, `PemError`
- Convenience macros: `invalid_arg!`, `io_error!`, `system_time_error!`, `invalid_op!`, `pem_error!`

#### `util`

| Module | bc-csharp source | Notes |
|--------|-----------------|-------|
| `util::integers` | `Integers.cs` | Bit manipulation for `u32` |
| `util::longs` | `Longs.cs` | Bit manipulation for `u64` |
| `util::shorts` | `Shorts.cs` | Bit manipulation for `u16` |
| `util::date` | `DateTimeUtilities.cs` | Unix timestamp utilities |
| `util::encoders::hex` | `HexEncoder.cs` | Hex encoding and decoding |
| `util::encoders::base64` | `Base64Encoder.cs`, `UrlBase64Encoder.cs` | Base64 (Standard, URL-safe, bc-csharp compatible) |
| `util::encoders::encoder` | `IEncoder.cs` | `Encoder` trait |
| `util::io::streams` | `Streams.cs` | Stream utility functions |
| `util::io::limited_reader` | `LimitedInputStream.cs` | Byte-limited `Read` wrapper |
| `util::io::pushback_reader` | `PushbackStream.cs` | Single-byte pushback `Read` wrapper |
| `util::io::tee_reader` | `TeeInputStream.cs` | `Read` wrapper that copies to secondary `Write` |
| `util::io::tee_writer` | `TeeOutputStream.cs` | `Write` wrapper that copies to secondary `Write` |
| `util::io::pem` | `util/io/pem/` | PEM encoding and decoding |

### Planned

- [ ] Hash functions (SHA-256, SHA-512, ...)
- [ ] Symmetric encryption (AES, ChaCha20, ...)
- [ ] Asymmetric encryption (RSA, ECDSA, ...)
- [ ] Key derivation (PBKDF2, ...)

## Building

```bash
cargo build
```

## Testing

```bash
cargo test
```

## Reference

- [bc-csharp source](https://github.com/bcgit/bc-csharp)
- [The Rust Programming Language](https://doc.rust-lang.org/book/)
