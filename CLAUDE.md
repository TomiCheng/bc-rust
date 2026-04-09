# bc-rust

## Project Goal

Port [bc-csharp](https://github.com/bcgit/bc-csharp) (Bouncy Castle C#) cryptography library to Rust as a learning exercise for both Rust and cryptographic algorithms. The goal is to publish it on crates.io for Rust developers.

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Production releases — published to crates.io |
| `develop` | Active development — merge feature branches here |

Feature branches should be cut from `develop` and merged back via PR.
`develop` is merged into `main` only when ready to publish a new release.

## Code Quality

After completing any code changes, run the full check suite and fix all issues before committing:

```
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```
