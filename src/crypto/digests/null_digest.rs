use std::fmt::{Display};
use crate::BcError;
use crate::crypto::Digest;
use crate::util::Memorable;

#[derive(Clone)]
pub struct NullDigest {
    buffer: Vec<u8>,
}
impl Display for NullDigest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.algorithm_name())
    }
}
impl Digest for NullDigest {
    fn algorithm_name(&self) -> String {
        "NULL".to_string()
    }

    fn digest_size(&self) -> usize {
        0
    }

    fn byte_length(&self) -> usize {
        self.buffer.len()
    }

    fn reset(&mut self) {
        self.buffer.clear();
    }

    fn update(&mut self, input: u8) -> Result<(), BcError> {
        self.buffer.push(input);
        Ok(())
    }

    fn block_update(&mut self, input: &[u8]) -> Result<(), BcError> {
        self.buffer.extend_from_slice(input);
        Ok(())
    }

    fn do_final(&mut self, output: &mut [u8]) -> Result<usize, BcError> {
        let len = self.buffer.len().min(output.len());
        output[..len].copy_from_slice(&self.buffer[..len]);
        self.reset();
        Ok(len)
    }
}
impl Memorable for NullDigest {
    fn restore(&mut self, other: &Self) -> Result<(), BcError> {
        self.buffer = other.buffer.clone();
        Ok(())
    }
}