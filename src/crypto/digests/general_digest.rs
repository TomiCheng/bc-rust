//! base implementation of MD4 family style digest as outlined in
//! "Handbook of Applied Cryptography", pages 344 - 347.

const BYTE_LENGTH: usize = 64;

#[derive(Clone)]
pub(crate) struct GeneralDigest<TDigiestImpl: InternalGeneralDigest + Clone> {
    x_buf: [u8; 4],
    x_buf_off: usize,
    byte_count: usize,
    digest_impl: TDigiestImpl,
}

impl<TDigiestImpl: InternalGeneralDigest + Clone> GeneralDigest<TDigiestImpl> {
    pub(crate) fn new(digest_impl: TDigiestImpl) -> GeneralDigest<TDigiestImpl> {
        GeneralDigest {
            x_buf: [0; 4],
            x_buf_off: 0,
            byte_count: 0,
            digest_impl
        }
    }
    
    pub(crate) fn update(&mut self, input: u8) {
        self.x_buf[self.x_buf_off] = input;
        self.x_buf_off += 1;
        if self.x_buf_off == self.x_buf.len() {
            self.digest_impl.process_word(&self.x_buf);
            self.x_buf_off = 0;
        }
        self.byte_count += 1;
    }

    pub(crate) fn block_update(&mut self, input: &[u8]) {
        let length = input.len();
        // fill the current word
        let mut i = 0;
        if self.x_buf_off > 0 {
            while i < length {
                self.x_buf[self.x_buf_off] = input[i]; 
                self.x_buf_off += 1;
                i += 1;
                if self.x_buf_off == 4 {
                    self.digest_impl.process_word(&self.x_buf);
                    self.x_buf_off = 0;
                    break;
                }
            }
        }

        // process whole words.
        let limit = length as isize - 3;
        while (i as isize) < limit {
            self.digest_impl.process_word(&input[i..]);
            i += 4;
        }

        // load in the remainder.
        while i < length {
            self.x_buf[self.x_buf_off] = input[i];
            self.x_buf_off += 1;
            i += 1;
        }

        self.byte_count += length;
    }

    pub(crate) fn finish(&mut self) {
        let bit_length = self.byte_count << 3; 
        
        // add the pad bytes.
        self.update(128);

        while self.x_buf_off != 0 {
            self.update(0);
        }

        self.digest_impl.process_length(bit_length);
        self.digest_impl.process_block();
    }

    pub(crate) fn reset(&mut self) {
        self.byte_count = 0;
        self.x_buf_off = 0;
        self.x_buf.fill(0);
    }

    pub(crate) fn get_byte_length(&self) -> usize {
        BYTE_LENGTH
    }

    pub(crate) fn as_impl_mut(&mut self) -> &mut TDigiestImpl {
        &mut self.digest_impl
    }
} 

pub(crate) trait InternalGeneralDigest {
    fn process_word(&mut self, word: &[u8]);
    fn process_length(&mut self, bit_length: usize);
    fn process_block(&mut self);
}

