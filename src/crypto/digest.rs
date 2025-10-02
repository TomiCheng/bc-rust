use crate::BcError;

/// Trait representing a generic cryptographic digest (hash) function.
pub trait Digest {
    /// The algorithm name
    fn algorithm_name(&self) -> String;
    /// The digest produced by this message digest.
    fn digest_size(&self) -> usize;
    /// The internal buffer used by this digest.
    fn byte_length(&self) -> usize;

    /// Reset the digest back to its initial state.
    fn reset(&mut self);
    /// Update the message digest with a single byte.
    fn update(&mut self, input: u8) -> Result<(), BcError>;
    /// Update the message digest with a span of bytes.
    fn block_update(&mut self, input: &[u8]) -> Result<(), BcError>;
    /// Close the digest, producing the final digest value.
    fn do_final(&mut self, output: &mut [u8]) -> Result<usize, BcError>;
}