use crate::crypto::Digest;
use crate::util::pack::{be_to_u32, u32_to_be_bytes};
use crate::util::Memoable;
use super::general_digest::{GeneralDigest, InternalGeneralDigest};

const Y1: u32 = 0x5a827999;
const Y2: u32 = 0x6ed9eba1;
const Y3: u32 = 0x8f1bbcdc;
const Y4: u32 = 0xca62c1d6;
const DIGEST_LENGTH: usize = 20;

/// implementation of SHA-1 as outlined in "Handbook of Applied Cryptography", pages 346 - 349.  
/// 
/// It is interesting to ponder why the, apart from the extra IV, the other difference here from MD5
/// is the "endianness" of the word processing!
#[derive(Clone)]
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
    fn get_algorithm_name(&self) -> &'static str {
        "SHA-1"
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

impl Memoable for Sha1Digest {
    fn copy(&self) -> Sha1Digest {
        self.clone()
    }

    fn reset(&mut self, other: &Self) {
        other.clone_into(self);
    }
}

#[derive(Clone)]
struct Sha1DigestImpl {
    x: [u32; 80],
    x_off: usize,
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
            x_off: 0,
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
        self.x_off = 0;
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

impl InternalGeneralDigest for Sha1DigestImpl {
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

        self.x_off = 0;
        self.x[0..16].fill(0);
    }
}

fn g(u: u32, v: u32, w: u32) -> u32 {
    (u & v) | (u & w) | (v & w)
}

fn h(u: u32, v: u32, w: u32) -> u32 {
    u ^ v ^ w
}

fn f(u: u32, v: u32, w: u32) -> u32 {
    (u & v) | (!u & w)
}
