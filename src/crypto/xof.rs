use super::Digest;

/// With FIPS PUB 202, a new kind of message digest was announced which supported extendable output or variable digest sizes.
/// This interface provides the extra methods required to support variable output on a digest implementation.
pub trait Xof: Digest {
    /// Output the results of the final calculation for this XOF to fill the output span.
    ///
    /// # Arguments
    /// * `output` - span to fill with the output bytes.
    ///
    /// # Returns
    /// the number of bytes written
    fn output_final(&mut self, output: &mut [u8]) -> usize;
    /// Start outputting the results of the final calculation for this XOF. Unlike OutputFinal, this method
    /// will continue producing output until the XOF is explicitly reset, or signals otherwise.
    ///
    /// # Arguments
    /// * `output` - span to fill with the output bytes.
    ///
    /// # Returns
    /// the number of bytes written
    fn output(&mut self, output: &mut [u8]) -> usize;
}
