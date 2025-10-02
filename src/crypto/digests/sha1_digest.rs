use std::fmt::{Display, Formatter};
use crate::BcError;
use crate::crypto::digest::Digest;
use crate::crypto::digests::general_digest::{DigestImpl, GeneralDigest};
use crate::crypto::util::pack::Pack;
use crate::util::Memorable;

#[derive(Debug, Clone)]
pub struct Sha1Digest {
    m: GeneralDigest<Sha1DigestImpl>,
}
impl Sha1Digest {
    pub fn new() -> Self {
        Self {
            m: GeneralDigest::new(Sha1DigestImpl::new()),
        }
    }
}
impl Display for Sha1Digest {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.algorithm_name())
    }
}
impl Digest for Sha1Digest {
    fn algorithm_name(&self) -> String {
        "SHA-1".to_string()
    }

    fn digest_size(&self) -> usize {
        DIGEST_LENGTH
    }

    fn byte_length(&self) -> usize {
        self.m.byte_length()
    }

    fn reset(&mut self) {
        self.m.reset();
        self.m.m.reset();
    }

    fn update(&mut self, input: u8) -> Result<(), BcError> {
        self.m.update(input);
        Ok(())
    }

    fn block_update(&mut self, input: &[u8]) -> Result<(), BcError> {
        self.m.block_update(input);
        Ok(())
    }

    fn do_final(&mut self, output: &mut [u8]) -> Result<usize, BcError> {
        self.m.finish();
        let size = self.m.m.do_final(output);
        self.reset();
        Ok(size)
    }
}
impl Memorable for Sha1Digest {
    fn restore(&mut self, other: &Self) -> Result<(), BcError> {
        self.m.restore(&other.m)
    }
}
#[derive(Debug, Clone)]
struct Sha1DigestImpl {
    pub x: [u32; 80],
    pub x_offset: usize,
    pub h1: u32,
    pub h2: u32,
    pub h3: u32,
    pub h4: u32,
    pub h5: u32,
}
impl Sha1DigestImpl {
    pub fn new() -> Self {
        let mut d = Self {
            x: [0; 80],
            x_offset: 0,
            h1: 0,
            h2: 0,
            h3: 0,
            h4: 0,
            h5: 0,
        };
        d.reset();
        d
    }
    fn reset(&mut self) {
        self.x.fill(0);
        self.x_offset = 0;
        self.h1 = 0x67452301;
        self.h2 = 0xefcdab89;
        self.h3 = 0x98badcfe;
        self.h4 = 0x10325476;
        self.h5 = 0xc3d2e1f0;
    }
    fn do_final(&mut self, output: &mut [u8]) -> usize {
        self.h1.to_be_slice(&mut output[0..4]);
        self.h2.to_be_slice(&mut output[4..8]);
        self.h3.to_be_slice(&mut output[8..12]);
        self.h4.to_be_slice(&mut output[12..16]);
        self.h5.to_be_slice(&mut output[16..20]);
        DIGEST_LENGTH
    }
}
impl DigestImpl for Sha1DigestImpl {
    fn process_word(&mut self, word: &[u8]) {
        self.x[self.x_offset] = u32::from_be_slice(&word[0..4]);
        if {
            self.x_offset += 1;
            self.x_offset
        } == 16
        {
            self.process_block();
        }
    }

    fn process_length(&mut self, bit_length: usize) {
        if self.x_offset > 14 {
            self.process_block();
        }
        self.x[14] = (bit_length as u64 >> 32) as u32;
        self.x[15] = (bit_length as u64) as u32;
    }

    fn process_block(&mut self) {
        // expand 16 word blocks into 80 word blocks.
        for i in 16..80 {
            let t = self.x[i - 3] ^ self.x[i - 8] ^ self.x[i - 14] ^ self.x[i - 16];
            self.x[i] = t.rotate_left(1);
        }

        // set up working variables.
        let mut a = self.h1;
        let mut b = self.h2;
        let mut c = self.h3;
        let mut d = self.h4;
        let mut e = self.h5;

        // round 1
        let mut idx = 0;
        for _ in 0..4 {
            e = e.wrapping_add(a.rotate_left(5).wrapping_add(f(b, c, d)).wrapping_add(self.x[idx]).wrapping_add(Y1));
            idx += 1;
            b = b.rotate_left(30);

            d = d.wrapping_add(e.rotate_left(5).wrapping_add(f(a, b, c)).wrapping_add(self.x[idx]).wrapping_add(Y1));
            idx += 1;
            a = a.rotate_left(30);

            c = c.wrapping_add(d.rotate_left(5).wrapping_add(f(e, a, b)).wrapping_add(self.x[idx]).wrapping_add(Y1));
            idx += 1;
            e = e.rotate_left(30);

            b = b.wrapping_add(c.rotate_left(5).wrapping_add(f(d, e, a)).wrapping_add(self.x[idx]).wrapping_add(Y1));
            idx += 1;
            d = d.rotate_left(30);

            a = a.wrapping_add(b.rotate_left(5).wrapping_add(f(c, d, e)).wrapping_add(self.x[idx]).wrapping_add(Y1));
            idx += 1;
            c = c.rotate_left(30);
        }

        // round 2
        for _ in 0..4 {
            e = e.wrapping_add(a.rotate_left(5).wrapping_add(h(b, c, d)).wrapping_add(self.x[idx]).wrapping_add(Y2));
            idx += 1;
            b = b.rotate_left(30);

            d = d.wrapping_add(e.rotate_left(5).wrapping_add(h(a, b, c)).wrapping_add(self.x[idx]).wrapping_add(Y2));
            idx += 1;
            a = a.rotate_left(30);

            c = c.wrapping_add(d.rotate_left(5).wrapping_add(h(e, a, b)).wrapping_add(self.x[idx]).wrapping_add(Y2));
            idx += 1;
            e = e.rotate_left(30);

            b = b.wrapping_add(c.rotate_left(5).wrapping_add(h(d, e, a)).wrapping_add(self.x[idx]).wrapping_add(Y2));
            idx += 1;
            d = d.rotate_left(30);

            a = a.wrapping_add(b.rotate_left(5).wrapping_add(h(c, d, e)).wrapping_add(self.x[idx]).wrapping_add(Y2));
            idx += 1;
            c = c.rotate_left(30);
        }

        // round 3
        for _ in 0..4 {
            e = e.wrapping_add(a.rotate_left(5).wrapping_add(g(b, c, d)).wrapping_add(self.x[idx]).wrapping_add(Y3));
            idx += 1;
            b = b.rotate_left(30);

            d = d.wrapping_add(e.rotate_left(5).wrapping_add(g(a, b, c)).wrapping_add(self.x[idx]).wrapping_add(Y3));
            idx += 1;
            a = a.rotate_left(30);

            c = c.wrapping_add(d.rotate_left(5).wrapping_add(g(e, a, b)).wrapping_add(self.x[idx]).wrapping_add(Y3));
            idx += 1;
            e = e.rotate_left(30);

            b = b.wrapping_add(c.rotate_left(5).wrapping_add(g(d, e, a)).wrapping_add(self.x[idx]).wrapping_add(Y3));
            idx += 1;
            d = d.rotate_left(30);

            a = a.wrapping_add(b.rotate_left(5).wrapping_add(g(c, d, e)).wrapping_add(self.x[idx]).wrapping_add(Y3));
            idx += 1;
            c = c.rotate_left(30);
        }

        // round 4
        for _ in 0..4 {
            e = e.wrapping_add(a.rotate_left(5).wrapping_add(h(b, c, d)).wrapping_add(self.x[idx]).wrapping_add(Y4));
            idx += 1;
            b = b.rotate_left(30);

            d = d.wrapping_add(e.rotate_left(5).wrapping_add(h(a, b, c)).wrapping_add(self.x[idx]).wrapping_add(Y4));
            idx += 1;
            a = a.rotate_left(30);

            c = c.wrapping_add(d.rotate_left(5).wrapping_add(h(e, a, b)).wrapping_add(self.x[idx]).wrapping_add(Y4));
            idx += 1;
            e = e.rotate_left(30);

            b = b.wrapping_add(c.rotate_left(5).wrapping_add(h(d, e, a)).wrapping_add(self.x[idx]).wrapping_add(Y4));
            idx += 1;
            d = d.rotate_left(30);

            a = a.wrapping_add(b.rotate_left(5).wrapping_add(h(c, d, e)).wrapping_add(self.x[idx]).wrapping_add(Y4));
            idx += 1;
            c = c.rotate_left(30);
        }

        self.h1 = self.h1.wrapping_add(a);
        self.h2 = self.h2.wrapping_add(b);
        self.h3 = self.h3.wrapping_add(c);
        self.h4 = self.h4.wrapping_add(d);
        self.h5 = self.h5.wrapping_add(e);

        self.x_offset = 0;
        self.x[0..16].fill(0);
    }
}
impl Memorable for Sha1DigestImpl {
    fn restore(&mut self, other: &Self) -> Result<(), BcError> {
        self.x = other.x;
        self.x_offset = other.x_offset;
        self.h1 = other.h1;
        self.h2 = other.h2;
        self.h3 = other.h3;
        self.h4 = other.h4;
        self.h5 = other.h5;
        Ok(())
    }
}

const DIGEST_LENGTH: usize = 20;
const Y1: u32 = 0x5a827999;
const Y2: u32 = 0x6ed9eba1;
const Y3: u32 = 0x8f1bbcdc;
const Y4: u32 = 0xca62c1d6;

fn g(u: u32, v: u32, w: u32) -> u32 {
    (u & v) | (u & w) | (v & w)
}
fn h(u: u32, v: u32, w: u32) -> u32 {
    u ^ v ^ w
}
fn f(u: u32, v: u32, w: u32) -> u32 {
    (u & v) | (!u & w)
}

#[cfg(test)]
mod tests {
    use crate::crypto::digests::test_digest::test_digest;
    use super::*;

    #[test]
    fn test() {
        let messages = ["", "a", "abc", "abcdefghijklmnopqrstuvwxyz"];

        let digests = [
            "da39a3ee5e6b4b0d3255bfef95601890afd80709",
            "86f7e437faa5a7fce15d1ddcb9eaeaea377667b8",
            "a9993e364706816aba3e25717850c26c9cd0d89d",
            "32d10c7b8cf96570ca04ce37f2a19d84240d3a89",
        ];

        test_digest(Sha1Digest::new(), &messages, &digests);
    }
}
