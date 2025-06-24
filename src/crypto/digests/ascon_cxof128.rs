use crate::{BcError, Result};
use crate::crypto::{Digest, Xof};
use crate::util::pack::{le_to_u64, u64_to_le};

const RATE: usize = 8;
/// Ascon-CXOF128 was introduced in NIST Special Publication (SP) 800-232 (Initial Public Draft).
///
/// # Remarks
/// Additional details and the specification can be found in:
/// [NIST SP 800-232 (Initial Public Draft)](https://csrc.nist.gov/pubs/sp/800/232/ipd)
/// For reference source code and implementation details, please see:
/// [Reference, highly optimized, masked C and ASM implementations of Ascon (NIST SP 800-232)](https://github.com/ascon/ascon-c)
pub struct AsconCxof128 {
    buffer: [u8; 8],
    z0: u64,
    z1: u64,
    z2: u64,
    z3: u64,
    z4: u64,
    s0: u64,
    s1: u64,
    s2: u64,
    s3: u64,
    s4: u64,
    buffer_index: usize,
    squeezing: bool
}

impl AsconCxof128 {
    pub fn new() -> Self {
        let mut result = Self {
            buffer: [0; 8],
            z0: 0,
            z1: 0,
            z2: 0,
            z3: 0,
            z4: 0,
            s0: 0,
            s1: 0,
            s2: 0,
            s3: 0,
            s4: 0,
            buffer_index: 0,
            squeezing: false
        };
        result.reset();
        result
    }
    pub fn with_init_state(z: &[u8]) -> Result<Self> {
        if z.len() > 256 {
            return Err(BcError::with_argument_out_of_range("customization string too long"));
        }

        let mut result = Self::new();
        result.init_state(z)?;
        result.z0 = result.s0;
        result.z1 = result.s1;
        result.z2 = result.s2;
        result.z3 = result.s3;
        result.z4 = result.s4;
        Ok(result)
    }
    fn init_state(&mut self, z: &[u8]) -> Result<()> {
        if z.len() == 0 {
            self.s0 = 0x500cccc894e3c9e8;
            self.s1 = 0x5bed06f28f71248d;
            self.s2 = 0x3b03a0f930afd512;
            self.s3 = 0x112ef093aa5c698b;
            self.s4 = 0x00c8356340a347f0;
        } else {
            self.s0 = 0x675527c2a0e8de03;
            self.s1 = 0x43d12d7dc0377bbc;
            self.s2 = 0xe9901dec426e81b5;
            self.s3 = 0x2ab14907720780b6;
            self.s4 = 0x8f3f1d02d432bc46;

            let bit_length = z.len() << 3;
            self.s0 ^= bit_length as u64;

            self.p12();
            self.block_update(z)?;
            self.pad_and_absorb();
            self.p12();
        }
        self.buffer_index = 0;
        Ok(())
    }
    fn p12(&mut self) {
        self.round(0xf0);
        self.round(0xe1);
        self.round(0xd2);
        self.round(0xc3);
        self.round(0xb4);
        self.round(0xa5);
        self.round(0x96);
        self.round(0x87);
        self.round(0x78);
        self.round(0x69);
        self.round(0x5a);
        self.round(0x4b);
    }
    #[inline]
    fn round(&mut self, c: u64) {
        let sx = self.s2 ^ c;
        let t0 = self.s0 ^ self.s1 ^ sx ^ self.s3 ^ (self.s1 & (self.s0 ^ sx ^ self.s4));
        let t1 = self.s0 ^ sx ^ self.s3 ^ self.s4 ^ ((self.s1 ^ sx) & (self.s1 ^ self.s3));
        let t2 = self.s1 ^ sx ^ self.s4 ^ (self.s3 & self.s4);
        let t3 = self.s0 ^ self.s1 ^ sx ^ (!self.s0 & (self.s3 ^ self.s4));
        let t4 = self.s1 ^ self.s3 ^ self.s4 ^ ((self.s0 ^ self.s4) & self.s1);

        self.s0 = t0 ^ t0.rotate_right(19) ^ t0.rotate_right(28);
        self.s1 = t1 ^ t1.rotate_right(39) ^ t1.rotate_right(61);
        self.s2 = !(t2 ^ t2.rotate_right(1) ^ t2.rotate_right(6));
        self.s3 = t3 ^ t3.rotate_right(10) ^ t3.rotate_right(17);
        self.s4 = t4 ^ t4.rotate_right(7) ^ t4.rotate_right(41);
    }
    fn pad_and_absorb(&mut self) {
        let final_bits = self.buffer_index << 3;
        self.s0 ^= le_to_u64(&self.buffer[0..8]) & (0x00FFFFFFFFFFFFFF >> (56 - final_bits));
        self.s0 ^= 0x01u64 << final_bits;
    }
}
impl Digest for AsconCxof128 {
    fn algorithm_name(&self) -> String {
        "Ascon-CXOF128".to_string()
    }

    fn get_digest_size(&self) -> usize {
        32
    }

    fn get_byte_length(&self) -> usize {
        RATE
    }

    fn update(&mut self, input: u8) -> Result<()> {
        if self.squeezing {
            return Err(BcError::with_invalid_operation("attempt to absorb while squeezing"));
        }
        self.buffer[self.buffer_index] = input;
        if { self.buffer_index += 1; self.buffer_index } == RATE {
            self.s0 ^= le_to_u64(&self.buffer[0..8]);
            self.p12();
            self.buffer_index = 0;
        }
        Ok(())
    }

    fn block_update(&mut self, input: &[u8]) -> Result<()> {
        if self.squeezing {
            return Err(BcError::with_invalid_operation("attempt to absorb while squeezing"));
        }

        let mut input = input;

        let available = RATE - self.buffer_index;
        if input.len() < available {
            self.buffer[self.buffer_index..(self.buffer_index + input.len())].copy_from_slice(input);
            self.buffer_index += input.len();
            return Ok(());
        }

        if self.buffer_index > 0 {
            let i = &input[..available];
            self.buffer[self.buffer_index..(self.buffer_index + i.len())].copy_from_slice(i);
            self.s0 ^= le_to_u64(&self.buffer[0..8]);
            input = &input[available..];
            self.p12();
        }

        while input.len() >= RATE {
            self.s0 ^= le_to_u64(&input[0..8]);
            input = &input[RATE..];
            self.p12();
        }

        self.buffer[0..input.len()].copy_from_slice(input);
        self.buffer_index = input.len();
        Ok(())
    }

    fn do_final(&mut self, output: &mut [u8]) -> Result<usize> {
        let digest_size = self.get_digest_size();
        if output.len() < digest_size {
            return Err(BcError::with_output_length("output buffer too small"))
        }
        Ok(self.output_final(output))
    }
    fn reset(&mut self) {
        self.s0 = self.z0;
        self.s1 = self.z1;
        self.s2 = self.z2;
        self.s3 = self.z3;
        self.s4 = self.z4;

        self.buffer.fill(0);
        self.buffer_index = 0;
        self.squeezing = false;
    }
}
impl Xof for AsconCxof128 {
    fn output_final(&mut self, output: &mut [u8]) -> usize {
        let length = self.output(output);
        self.reset();
        length
    }

    fn output(&mut self, output: &mut [u8]) -> usize {
        let mut output = output;

        let result = output.len();
        if !self.squeezing {
            self.pad_and_absorb();
            self.squeezing = true;
            self.buffer_index = 8;
        } else if self.buffer_index < 8 {
            let available = 8 - self.buffer_index;
            if output.len() <= available {
                output.copy_from_slice(&self.buffer[self.buffer_index..(self.buffer_index + output.len())]);
                self.buffer_index += output.len();
                return output.len();
            }

            output[..available].copy_from_slice(&self.buffer[self.buffer_index..(self.buffer_index + available)]);
            output = &mut output[available..];
            self.buffer_index = 8;
        }

        while output.len() >= 8 {
            self.p12();
            u64_to_le(self.s0, output);
            output = &mut output[8..];
        }

        if output.len() > 0 {
            self.p12();
            u64_to_le(self.s0, &mut self.buffer);
            output.copy_from_slice(&self.buffer[0..output.len()]);
            self.buffer_index = output.len();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::random::{random, DefaultRandomSource, RandomSource};
    use crate::crypto::digests::ascon_cxof128::AsconCxof128;
    use crate::crypto::digests::test_digest;
    use crate::crypto::Xof;
    use crate::util::encoders::hex::to_decode_with_str;

    #[test]
    fn test_output_xof_ascon_cxof128() {
        impl_test_output_xof(AsconCxof128::new());
    }
    #[test]
    fn bench_xof_ascon_cxof128() {
        impl_bench_xof(AsconCxof128::new());
    }
    #[test]
    fn test_digest_reset() {
        check_digest_reset(AsconCxof128::new());
    }
    #[test]
    fn test_vectors_xof_ascon_xof128() {
        let mut random_source = DefaultRandomSource::default();
        let file = File::open("./src/crypto/digests/LWC_CXOF_KAT_128_512.txt").unwrap();
        let mut reader = BufReader::new(file);

        let mut line = String::new();
        while reader.read_line(&mut line).unwrap() > 0 {
            let count: u32 = line.split('=').nth(1).unwrap().trim().parse().unwrap();
            line.clear();
            reader.read_line(&mut line).unwrap();
            let msg = line.split('=').nth(1).unwrap().trim();
            let msg_buffer = to_decode_with_str(msg).unwrap();
            line.clear();
            reader.read_line(&mut line).unwrap();
            let z = line.split('=').nth(1).unwrap().trim();
            let z_buffer = to_decode_with_str(z).unwrap();
            line.clear();
            reader.read_line(&mut line).unwrap();
            let md = line.split('=').nth(1).unwrap().trim();
            let md_buffer = to_decode_with_str(md).unwrap();
            line.clear();
            reader.read_line(&mut line).unwrap();

            let mut ascon = AsconCxof128::with_init_state(&z_buffer).unwrap();
            impl_test_vector_xof(&mut random_source, &mut ascon, count, &msg_buffer, &md_buffer);

            line.clear();
        }
    }
    fn impl_test_output_xof<TXof: Xof>(mut xof: TXof) {
        let mut expected = [0u8; 64];
        xof.output_final(&mut expected);
        let mut random_source = DefaultRandomSource::default();
        let output = &mut [0u8; 64];
        for i in 0..64 {
            random_source.fill_bytes(output);

            let mut pos = 0;
            while pos <= (output.len() - 16) {
                let len: usize = random::<usize>() % 17;
                xof.output(&mut output[pos..(pos + len)]);
                pos += len;
            }

            let remaining = output.len() - pos;
            xof.output_final(&mut output[pos..(pos + remaining)]);

            assert_eq!(expected, *output, "XOF output mismatch at iteration {}", i);
        }
    }
    fn impl_bench_xof<TXof: Xof>(mut xof: TXof) {
        let mut data = vec![0u8; 1024];
        for _ in 0..1024 {
            for _ in 0..1024 {
                xof.block_update(&data).unwrap()
            }
            xof.output_final(&mut data);
        }
    }
    fn impl_test_vector_xof<TXof: Xof>(random_source: &mut DefaultRandomSource,
                                       xof: &mut TXof,
                                       count: u32,
                                       msg: &[u8],
                                       expected: &[u8]) {
        let output_length = expected.len();
        let mut output = vec![0u8; output_length];
        {
            random_source.fill_bytes(&mut output);

            xof.block_update(msg).unwrap();
            xof.output_final(&mut output);
            assert_eq!(expected, output, "count: {}", count);
        }
        if msg.len() > 1 {
            random_source.fill_bytes(&mut output);
            let split_output: usize = random::<usize>() % output_length;
            xof.block_update(&msg).unwrap();
            xof.output(&mut output[..split_output]);
            xof.output_final(&mut output[split_output..]);
            assert_eq!(expected, output, "count: {}", count);
        }
    }
    fn check_digest_reset<TXof: Xof>(xof: TXof) {
        assert!(test_digest::test_digest_reset(xof));
    }
}

