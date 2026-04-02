# bc-rust

A Rust port of [Bouncy Castle C#](https://github.com/bcgit/bc-csharp) — a comprehensive cryptography library.

This is a learning project to explore Rust through practical implementation of cryptographic algorithms.

> **Note:** This project is developed with AI-assisted code generation using [Claude](https://claude.ai/). It serves as both a Rust learning exercise and an exploration of AI-assisted development workflows.

## Goals

- Port core cryptographic algorithms from bc-csharp to idiomatic Rust
- Learn Rust ownership, traits, generics, and error handling through real-world code

## Status

### Completed

- `error` — library-wide error type and macros
- `util` — utility modules (integers, encoders, I/O, PEM, date)

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
