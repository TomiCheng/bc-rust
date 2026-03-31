# bc-rust

A Rust port of [Bouncy Castle C#](https://github.com/bcgit/bc-csharp) — a comprehensive cryptography library.

This is a learning project to explore Rust through practical implementation of cryptographic algorithms.

## Goals

- Port core cryptographic algorithms from bc-csharp to idiomatic Rust
- Learn Rust ownership, traits, generics, and error handling through real-world code

## Planned Scope

- [ ] Error handling (`BcError`)
- [ ] Hash functions (SHA-256, SHA-512, MD5, ...)
- [ ] Symmetric encryption (AES, ChaCha20, ...)
- [ ] Asymmetric encryption (RSA, ECDSA, ...)
- [ ] Key derivation (PBKDF2, Argon2)

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
