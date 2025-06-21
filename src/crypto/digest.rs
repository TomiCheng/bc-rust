
/// Base trait for a message digest.
pub trait Digest {
    /// The algorithm name.
    fn algorithm_name(&self) -> String;
    /// Return the size, in bytes, of the digest produced by this message digest.
    ///
    /// # Returns
    /// The size, in bytes, of the digest produced by this message digest.
    fn get_digest_size(&self) -> usize;
    /// Return the size, in bytes, of the internal buffer used by this digest.
    ///
    /// # Returns
    /// The size, in bytes, of the internal buffer used by this digest.
    fn get_byte_length(&self) -> usize;
    /// Update the message digest with a single byte.
    ///
    /// # Arguments
    /// * `input` - The input byte to be entered.
    fn update(&mut self, input: u8);
    /// Update the message digest with a span of bytes.
    ///
    /// # Arguments
    /// * `input` - The span containing the data.
    fn block_update(&mut self, input: &[u8]);
    /// Close the digest, producing the final digest value.
    ///
    /// # Arguments
    /// * `output` - The span the digest is to be copied into.
    ///
    /// # Returns
    /// The number of bytes written.
    ///
    /// # Note
    /// This call leaves the digest reset.
    fn do_final(&mut self, output: &mut [u8]) -> usize;
    /// Reset the digest back to its initial state.
    fn reset(&mut self);
}