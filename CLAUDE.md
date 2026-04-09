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

## Development Workflow

For each new feature or fix:

1. Create a GitHub issue describing the work
2. Cut a branch from `develop`: `git checkout -b feature/issue-N-description`
3. Develop, then run the full check suite (see Code Quality below)
4. Push and open a PR targeting `develop`, referencing the issue with `Closes #N`
5. Merge PR into `develop`

## Porting Guidelines

When porting a bc-csharp class to Rust, not every method needs a direct implementation.
If Rust's standard library already provides an equivalent, **skip the implementation** and
document the alternative in the module-level doc comment instead.

## Documentation

Every `pub` function, struct, enum, and trait **must** have a doc comment (`///`).

- Use `///` for items, `//!` for module-level docs.
- Include an `# Examples` section with a runnable doctest where practical.
- Doc comments are enforced by `#![warn(missing_docs)]` in `lib.rs`.

## Code Quality

After completing any code changes, run the full check suite and fix all issues before committing:

```
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```
