use crate::Result;
use crate::crypto::Digest;
use crate::crypto::digests::general_digest::{DigestImpl, GeneralDigest};
use crate::util::Memoable;
use crate::util::pack::{be_to_u32, u32_to_be_bytes};

const DIGEST_LENGTH: usize = 20;

struct Sha1DigestImpl {
    x: [u32; 80],
    x_offset: usize,
    h1: u32,
    h2: u32,
    h3: u32,
    h4: u32,
    h5: u32,
}

impl Sha1DigestImpl {
    fn new() -> Sha1DigestImpl {
        let mut result = Sha1DigestImpl {
            x: [0; 80],
            x_offset: 0,
            h1: 0,
            h2: 0,
            h3: 0,
            h4: 0,
            h5: 0,
        };
        result.reset();
        result
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
        u32_to_be_bytes(self.h1, &mut output[0..4]);
        u32_to_be_bytes(self.h2, &mut output[4..8]);
        u32_to_be_bytes(self.h3, &mut output[8..12]);
        u32_to_be_bytes(self.h4, &mut output[12..16]);
        u32_to_be_bytes(self.h5, &mut output[16..20]);
        DIGEST_LENGTH
    }
}
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
impl DigestImpl for Sha1DigestImpl {
    fn process_word(&mut self, word: &[u8]) {
        self.x[self.x_offset] = be_to_u32(&word[0..4]);
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
impl Memoable for Sha1DigestImpl {
    fn copy(&self) -> Self {
        Sha1DigestImpl {
            x: self.x,
            x_offset: self.x_offset,
            h1: self.h1,
            h2: self.h2,
            h3: self.h3,
            h4: self.h4,
            h5: self.h5,
        }
    }

    fn restore(&mut self, other: &Self) -> Result<()> {
        self.x.copy_from_slice(&other.x);
        self.x_offset = other.x_offset;
        self.h1 = other.h1;
        self.h2 = other.h2;
        self.h3 = other.h3;
        self.h4 = other.h4;
        self.h5 = other.h5;
        Ok(())
    }
}

pub struct Sha1Digest {
    digest_impl: GeneralDigest<Sha1DigestImpl>,
}

impl Sha1Digest {
    pub fn new() -> Sha1Digest {
        Sha1Digest {
            digest_impl: GeneralDigest::new(Sha1DigestImpl::new()),
        }
    }
}
impl Digest for Sha1Digest {
    fn algorithm_name(&self) -> String {
        "SHA-1".to_string()
    }

    fn get_digest_size(&self) -> usize {
        DIGEST_LENGTH
    }

    fn get_byte_length(&self) -> usize {
        self.digest_impl.get_byte_length()
    }

    fn update(&mut self, input: u8) -> Result<()> {
        self.digest_impl.update(input);
        Ok(())
    }

    fn block_update(&mut self, input: &[u8]) -> Result<()> {
        self.digest_impl.block_update(input);
        Ok(())
    }

    fn do_final(&mut self, output: &mut [u8]) -> Result<usize> {
        self.digest_impl.finish();
        let size = self.digest_impl.as_impl_mut().do_final(output);
        Digest::reset(self);
        Ok(size)
    }

    fn reset(&mut self) {
        self.digest_impl.reset();
        self.digest_impl.as_impl_mut().reset();
    }
}
impl Memoable for Sha1Digest {
    fn copy(&self) -> Self {
        Sha1Digest {
            digest_impl: self.digest_impl.copy(),
        }
    }

    fn restore(&mut self, other: &Self) -> Result<()> {
        self.digest_impl.restore(&other.digest_impl)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::digests::test_digest::test_digest;

    #[test]
    fn test() {
        let messages = vec!["", "a", "abc", "abcdefghijklmnopqrstuvwxyz"];

        let digests = vec![
            "da39a3ee5e6b4b0d3255bfef95601890afd80709",
            "86f7e437faa5a7fce15d1ddcb9eaeaea377667b8",
            "a9993e364706816aba3e25717850c26c9cd0d89d",
            "32d10c7b8cf96570ca04ce37f2a19d84240d3a89",
        ];

        test_digest(Sha1Digest::new(), &messages, &digests);
    }
}
