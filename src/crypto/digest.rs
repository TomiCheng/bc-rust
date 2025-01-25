
/// Base interface for a message digest.
pub trait Digest {
    /// The algorithm name.
    fn get_algorythm_name(&self) -> &'static str;

    /// Return the size, in bytes, of the digest produced by this message digest.
    fn get_digest_size(&self) -> usize;

    /// Return the size, in bytes, of the internal buffer used by this digest.
    fn get_byte_length(&self) -> usize;

    /// Update the message digest with a single byte.
    fn update(&mut self, input: u8);

    /// Update the message digest with a span of bytes.
    fn block_update(&mut self, input: &[u8]);

    /// Close the digest, producing the final digest value.
    fn do_final(&mut self, output: &mut[u8]) -> usize;
}