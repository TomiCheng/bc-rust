use super::general_digest::{GeneralDigest, InternalGeneralDigest};
use crate::crypto::Digest;
use crate::util::pack::{be_to_u32, u32_to_be_bytes};
use crate::util::Memoable;

const DIGEST_LENGTH: usize = 32;

/// SHA-256 Constants (represent the first 32 bits of the fractional parts of the cube roots of the first sixty-four prime numbers)
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Draft FIPS 180-2 implementation of SHA-256. 
/// **Note:** As this is based on a draft this implementation is subject to change.
#[derive(Clone)]
pub struct Sha256Digest {
    digest_impl: GeneralDigest<Sha256DigestImpl>,
}

impl Sha256Digest {
    pub fn new() -> Sha256Digest {
        Sha256Digest {
            digest_impl: GeneralDigest::new(Sha256DigestImpl::new()),
        }
    }
}

impl Digest for Sha256Digest {
    fn get_algorithm_name(&self) -> &'static str {
        "SHA-256"
    }

    fn get_digest_size(&self) -> usize {
        DIGEST_LENGTH
    }

    fn get_byte_length(&self) -> usize {
        self.digest_impl.get_byte_length()
    }

    fn update(&mut self, input: u8) {
        self.digest_impl.update(input);
    }

    fn block_update(&mut self, input: &[u8]) {
        self.digest_impl.block_update(input);
    }

    fn do_final(&mut self, output: &mut [u8]) -> usize {
        self.digest_impl.finish();
        let size = self.digest_impl.as_impl_mut().do_final(output);
        Digest::reset(self);
        size
    }

    fn reset(&mut self) {
        self.digest_impl.reset();
        self.digest_impl.as_impl_mut().reset();
    }
}

impl Memoable for Sha256Digest {
    fn copy(&self) -> Sha256Digest {
        self.clone()
    }

    fn reset(&mut self, other: &Self) {
        other.clone_into(self);
    }
}

#[derive(Clone)]
struct Sha256DigestImpl {
    x: [u32; 64],
    x_off: usize,
    h1: u32,
    h2: u32,
    h3: u32,
    h4: u32,
    h5: u32,
    h6: u32,
    h7: u32,
    h8: u32,
}

impl Sha256DigestImpl {
    fn new() -> Sha256DigestImpl {
        let mut result = Sha256DigestImpl {
            x: [0; 64],
            x_off: 0,
            h1: 0,
            h2: 0,
            h3: 0,
            h4: 0,
            h5: 0,
            h6: 0,
            h7: 0,
            h8: 0,
        };
        result.reset();
        result
    }

    fn reset(&mut self) {
        // SHA-256 initial hash value The first 32 bits of the fractional parts of the square roots of the first eight prime numbers
        self.x.fill(0);
        self.x_off = 0;
        self.h1 = 0x6a09e667;
        self.h2 = 0xbb67ae85;
        self.h3 = 0x3c6ef372;
        self.h4 = 0xa54ff53a;
        self.h5 = 0x510e527f;
        self.h6 = 0x9b05688c;
        self.h7 = 0x1f83d9ab;
        self.h8 = 0x5be0cd19;
    }

    fn do_final(&mut self, output: &mut [u8]) -> usize {
        u32_to_be_bytes(self.h1, &mut output[0..4]);
        u32_to_be_bytes(self.h2, &mut output[4..8]);
        u32_to_be_bytes(self.h3, &mut output[8..12]);
        u32_to_be_bytes(self.h4, &mut output[12..16]);
        u32_to_be_bytes(self.h5, &mut output[16..20]);
        u32_to_be_bytes(self.h6, &mut output[20..24]);
        u32_to_be_bytes(self.h7, &mut output[24..28]);
        u32_to_be_bytes(self.h8, &mut output[28..32]);
        DIGEST_LENGTH
    }
}

impl InternalGeneralDigest for Sha256DigestImpl {
    fn process_word(&mut self, word: &[u8]) {
        self.x[self.x_off] = be_to_u32(&word[0..4]);
        if {
            self.x_off += 1;
            self.x_off
        } == 16
        {
            self.process_block();
        }
    }

    fn process_length(&mut self, bit_length: usize) {
        if self.x_off > 14 {
            self.process_block();
        }
        self.x[14] = (bit_length as u64 >> 32) as u32;
        self.x[15] = (bit_length as u64) as u32;
    }

    fn process_block(&mut self) {
        // expand 16 word block into 80 word block.
        for ti in 16..64 {
            self.x[ti] = theta1(self.x[ti - 2])
                .wrapping_add(self.x[ti - 7])
                .wrapping_add(theta0(self.x[ti - 15]))
                .wrapping_add(self.x[ti - 16]);
        }

        // set up working variables.
        let mut a = self.h1;
        let mut b = self.h2;
        let mut c = self.h3;
        let mut d = self.h4;
        let mut e = self.h5;
        let mut f = self.h6;
        let mut g = self.h7;
        let mut h = self.h8;

        let mut t = 0;
        for _ in 0..8 {
            // t = 8 * i
            h = h.wrapping_add(sum_l_ch(e, f, g).wrapping_add(K[t]).wrapping_add(self.x[t]));
            d = d.wrapping_add(h);
            h = h.wrapping_add(sum_o_maj(a, b, c));
            t += 1;

            // t = 8 * i + 1
            g = g.wrapping_add(sum_l_ch(d, e, f).wrapping_add(K[t]).wrapping_add(self.x[t]));
            c = c.wrapping_add(g);
            g = g.wrapping_add(sum_o_maj(h, a, b));
            t += 1;

            // t = 8 * i + 2
            f = f.wrapping_add(sum_l_ch(c, d, e).wrapping_add(K[t]).wrapping_add(self.x[t]));
            b = b.wrapping_add(f);
            f = f.wrapping_add(sum_o_maj(g, h, a));
            t += 1;

            // t = 8 * i + 3
            e = e.wrapping_add(sum_l_ch(b, c, d).wrapping_add(K[t]).wrapping_add(self.x[t]));
            a = a.wrapping_add(e);
            e = e.wrapping_add(sum_o_maj(f, g, h));
            t += 1;

            // t = 8 * i + 4
            d = d.wrapping_add(sum_l_ch(a, b, c).wrapping_add(K[t]).wrapping_add(self.x[t]));
            h = h.wrapping_add(d);
            d = d.wrapping_add(sum_o_maj(e, f, g));
            t += 1;

            // t = 8 * i + 5
            c = c.wrapping_add(sum_l_ch(h, a, b).wrapping_add(K[t]).wrapping_add(self.x[t]));
            g = g.wrapping_add(c);
            c = c.wrapping_add(sum_o_maj(d, e, f));
            t += 1;

            // t = 8 * i + 6
            b = b.wrapping_add(sum_l_ch(g, h, a).wrapping_add(K[t]).wrapping_add(self.x[t]));
            f = f.wrapping_add(b);
            b = b.wrapping_add(sum_o_maj(c, d, e));
            t += 1;

            // t = 8 * i + 7
            a = a.wrapping_add(sum_l_ch(f, g, h).wrapping_add(K[t]).wrapping_add(self.x[t]));
            e = e.wrapping_add(a);
            a = a.wrapping_add(sum_o_maj(b, c, d));
            t += 1;
        }

        self.h1 = self.h1.wrapping_add(a);
        self.h2 = self.h2.wrapping_add(b);
        self.h3 = self.h3.wrapping_add(c);
        self.h4 = self.h4.wrapping_add(d);
        self.h5 = self.h5.wrapping_add(e);
        self.h6 = self.h6.wrapping_add(f);
        self.h7 = self.h7.wrapping_add(g);
        self.h8 = self.h8.wrapping_add(h);

        self.x_off = 0;
        self.x[0..16].fill(0);
    }
}

fn sum_o_maj(x: u32, y: u32, z: u32) -> u32 {
    (x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)).wrapping_add((x & y) | (z & (x ^ y)))
}

fn sum_l_ch(x: u32, y: u32, z: u32) -> u32 {
    (x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)).wrapping_add(z ^ (x & (y ^ z)))
}

fn theta0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}

fn theta1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}
