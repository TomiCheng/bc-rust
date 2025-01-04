use std::any::Any;

/// Base interface for a symmetric key block cipher.
pub trait BlockCipher {
    /// Initialise the cipher.
    /// 
    /// # Arguments
    /// * `for_encryption` - Initialise for encryption if true, for decryption if false.
    /// * `parameters` - The key or other data required by the cipher.
    fn init(&mut self, for_encryption: bool, parameters: &dyn Any);
    /// The name of the algorithm this cipher implements.
    fn get_algorithm_name(&self) -> &str;
    /// The block size for this cipher, in bytes.
    fn get_block_size(&self) -> usize;
    /// Process a block.
    fn process_block(&self, input: &[u8], output: &mut [u8]) -> usize;
}