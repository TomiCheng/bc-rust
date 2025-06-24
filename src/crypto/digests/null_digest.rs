use crate::crypto::Digest;
use crate::Result;

#[derive(Clone)]
pub struct NullDigest {
    buffer: Vec<u8>
}

impl Digest for NullDigest {
    fn algorithm_name(&self) -> String {
        "NULL".to_string()
    }

    fn get_digest_size(&self) -> usize {
        0
    }

    fn get_byte_length(&self) -> usize {
        self.buffer.len()
    }

    fn update(&mut self, input: u8) -> Result<()> {
        self.buffer.push(input);
        Ok(())
    }

    fn block_update(&mut self, input: &[u8]) -> Result<()> {
        self.buffer.extend_from_slice(input);
        Ok(())
    }

    fn do_final(&mut self, output: &mut [u8]) -> Result<usize> {
        let len = self.buffer.len().min(output.len());
        output[..len].copy_from_slice(&self.buffer[..len]);
        self.reset();
        Ok(len)
    }

    fn reset(&mut self) {
        self.buffer.clear();
    }
}