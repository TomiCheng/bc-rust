//! base implementation of MD4 family style digest as outlined in
//! "Handbook of Applied Cryptography", pages 344-347.

use crate::Result;
use crate::util::Memoable;

const BYTE_LENGTH: usize = 64;

pub(crate) trait DigestImpl {
    fn process_word(&mut self, word: &[u8]);
    fn process_length(&mut self, bit_length: usize);
    fn process_block(&mut self);
}

pub(crate) struct GeneralDigest<TDigestImpl: DigestImpl + Memoable> {
    x: [u8; 4],
    x_offset: usize,
    byte_count: usize,
    digest_impl: TDigestImpl,
}

impl<TDigestImpl: DigestImpl + Memoable> GeneralDigest<TDigestImpl> {
    pub(crate) fn new(digest_impl: TDigestImpl) -> Self {
        Self {
            x: [0; 4],
            x_offset: 0,
            byte_count: 0,
            digest_impl,
        }
    }
    pub(crate) fn update(&mut self, input: u8) {
        self.x[self.x_offset] = input;
        self.x_offset += 1;

        if self.x_offset == self.x.len() {
            self.digest_impl.process_word(&self.x);
            self.x_offset = 0;
        }

        self.byte_count += 1;
    }
    pub(crate) fn block_update(&mut self, input: &[u8]) {
        if input.is_empty() {
            return;
        }

        let mut slice = input;

        // fill the current word
        while self.x_offset > 0 && slice.len() > 0 {
            self.x[self.x_offset] = slice[0];
            self.x_offset += 1;
            slice = &slice[1..];
            if self.x_offset == self.x.len() {
                self.digest_impl.process_word(&self.x);
                self.x_offset = 0;
                break;
            }
        }

        // process whole words.
        while slice.len() >= self.x.len() {
            self.digest_impl.process_word(&slice[..self.x.len()]);
            slice = &slice[self.x.len()..];
        }

        // load in the remainder.
        while slice.len() > 0 {
            self.x[self.x_offset] = slice[0];
            self.x_offset += 1;
            slice = &slice[1..];
            if self.x_offset == self.x.len() {
                self.digest_impl.process_word(&self.x);
                self.x_offset = 0;
            }
        }

        self.byte_count += input.len();
    }
    pub(crate) fn finish(&mut self) {
        let bit_length = self.byte_count << 3;
        // add the pad bytes.
        self.update(128);

        while self.x_offset != 0 {
            self.update(0);
        }

        self.digest_impl.process_length(bit_length);
        self.digest_impl.process_block();
    }
    pub(crate) fn reset(&mut self) {
        self.byte_count = 0;
        self.x_offset = 0;
        self.x.fill(0);
    }
    pub(crate) fn get_byte_length(&self) -> usize {
        BYTE_LENGTH
    }
    pub(crate) fn as_impl_mut(&mut self) -> &mut TDigestImpl {
        &mut self.digest_impl
    }
}

impl<TDigestImpl: DigestImpl + Memoable> Memoable for GeneralDigest<TDigestImpl> {
    fn copy(&self) -> Self {
        GeneralDigest {
            x: self.x,
            x_offset: self.x_offset,
            byte_count: self.byte_count,
            digest_impl: self.digest_impl.copy(),
        }
    }

    fn restore(&mut self, other: &Self) -> Result<()> {
        self.x.copy_from_slice(&other.x);
        self.x_offset = other.x_offset;
        self.byte_count = other.byte_count;
        self.digest_impl.restore(&other.digest_impl)
    }
}
