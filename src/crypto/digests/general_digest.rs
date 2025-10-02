//! base implementation of MD4 family style digest as outlined in
//! "Handbook of Applied Cryptography", pages 344-347.

use crate::BcError;
use crate::util::memorable::Memorable;

const BYTE_LENGTH: usize = 64;
pub(crate) trait DigestImpl {
    fn process_word(&mut self, word: &[u8]);
    fn process_length(&mut self, bit_length: usize);
    fn process_block(&mut self);
}

#[derive(Debug, Clone)]
pub(crate) struct GeneralDigest<Impl: DigestImpl> {
    pub x: [u8; 4],
    pub x_offset: usize,
    pub byte_count: usize,
    pub m: Impl,
}
impl<Impl: DigestImpl + Memorable> GeneralDigest<Impl> {
    pub(crate) fn new(m: Impl) -> Self {
        Self {
            x: [0; 4],
            x_offset: 0,
            byte_count: 0,
            m,
        }
    }
    pub(crate) fn update(&mut self, input: u8) {
        self.x[self.x_offset] = input;
        self.x_offset += 1;

        if self.x_offset == self.x.len() {
            self.m.process_word(self.x.as_ref());
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
                self.m.process_word(&self.x);
                self.x_offset = 0;
                break;
            }
        }

        // process whole words.
        while slice.len() >= self.x.len() {
            self.m.process_word(&slice[..self.x.len()]);
            slice = &slice[self.x.len()..];
        }

        // load in the remainder.
        while slice.len() > 0 {
            self.x[self.x_offset] = slice[0];
            self.x_offset += 1;
            slice = &slice[1..];
            if self.x_offset == self.x.len() {
                self.m.process_word(&self.x);
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

        self.m.process_length(bit_length);
        self.m.process_block();
    }
    pub(crate) fn reset(&mut self) {
        self.byte_count = 0;
        self.x_offset = 0;
        self.x.fill(0);
    }
    pub(crate) fn byte_length(&self) -> usize {
        BYTE_LENGTH
    }
}
impl<Impl: DigestImpl + Memorable> Memorable for GeneralDigest<Impl> {
    fn restore(&mut self, other: &Self) -> Result<(), BcError> {
        self.x.copy_from_slice(&other.x);
        self.x_offset = other.x_offset;
        self.byte_count = other.byte_count;
        self.m.restore(&other.m)
    }
}