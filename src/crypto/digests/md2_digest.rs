use std::fmt::{Display, Formatter};
use crate::BcError;
use crate::crypto::Digest;
use crate::util::Memorable;

#[derive(Clone)]
pub struct Md2Digest {
    x: [u8; 48],
    m: [u8; 16],
    c: [u8; 16],
    x_offset: usize,
    m_offset: usize,
    c_offset: usize,
}
impl Md2Digest {
    pub fn new() -> Self {
        Md2Digest {
            x: [0; 48],
            m: [0; 16],
            c: [0; 16],
            x_offset: 0,
            m_offset: 0,
            c_offset: 0,
        }
    }
}
impl Display for Md2Digest {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.algorithm_name())
    }
}
impl Digest for Md2Digest {
    fn algorithm_name(&self) -> String {
        "MD2".to_string()
    }

    fn digest_size(&self) -> usize {
        DIGEST_LENGTH
    }

    fn byte_length(&self) -> usize {
        BYTE_LENGTH
    }

    fn update(&mut self, input: u8) -> Result<(), BcError> {
        self.m[self.m_offset] = input;
        self.m_offset += 1;

        if self.m_offset == 16 {
            process_check_sum(&mut self.c, &self.m);
            process_block(&mut self.x, &self.m);
            self.m_offset = 0;
        }
        Ok(())
    }

    fn block_update(&mut self, input: &[u8]) -> Result<(), BcError> {
        let mut slice = input;

        // fill the current word
        while self.m_offset != 0 && slice.len() > 0 {
            self.update(slice[0])?;
            slice = &slice[1..];
        }

        // process whole words.
        while slice.len() >= BYTE_LENGTH {
            self.m.copy_from_slice(&slice[..16]);
            process_check_sum(&mut self.c, &self.m);
            process_block(&mut self.x, &self.m);
            slice = &slice[BYTE_LENGTH..];
        }

        // load in the remainder
        while slice.len() > 0 {
            self.update(slice[0])?;
            slice = &slice[1..];
        }

        Ok(())
    }

    fn do_final(&mut self, output: &mut [u8]) -> Result<usize, BcError> {
        // add padding
        let padding_byte = (self.m.len() - self.m_offset) as u8;
        for i in self.m_offset..self.m.len() {
            self.m[i] = padding_byte;
        }
        // do a final check sum
        process_check_sum(&mut self.c, &self.m);
        process_block(&mut self.x, &self.m);
        process_block(&mut self.x, &self.c);

        output.copy_from_slice(&self.x[self.x_offset..(self.x_offset + DIGEST_LENGTH)]);
        self.reset();
        Ok(DIGEST_LENGTH)
    }

    fn reset(&mut self) {
        self.m = [0; 16];
        self.c = [0; 16];
        self.x = [0; 48];
        self.x_offset = 0;
        self.m_offset = 0;
        self.c_offset = 0;
    }
}
impl Memorable for Md2Digest {
    fn restore(&mut self, other: &Self) -> Result<(), BcError> {
        self.x = other.x;
        self.m = other.m;
        self.c = other.c;
        self.x_offset = other.x_offset;
        self.m_offset = other.m_offset;
        self.c_offset = other.c_offset;
        Ok(())
    }
}
const DIGEST_LENGTH: usize = 16;
const BYTE_LENGTH: usize = 16;
/// 256-byte random permutation constructed from the digits of PI
static S: [u8; 256] = [
    41, 46, 67, 201, 162, 216, 124, 1, 61, 54, 84, 161, 236, 240, 6, 19, 98, 167, 5, 243, 192, 199, 115, 140, 152, 147, 43, 217, 188, 76, 130, 202,
    30, 155, 87, 60, 253, 212, 224, 22, 103, 66, 111, 24, 138, 23, 229, 18, 190, 78, 196, 214, 218, 158, 222, 73, 160, 251, 245, 142, 187, 47, 238,
    122, 169, 104, 121, 145, 21, 178, 7, 63, 148, 194, 16, 137, 11, 34, 95, 33, 128, 127, 93, 154, 90, 144, 50, 39, 53, 62, 204, 231, 191, 247, 151,
    3, 255, 25, 48, 179, 72, 165, 181, 209, 215, 94, 146, 42, 172, 86, 170, 198, 79, 184, 56, 210, 150, 164, 125, 182, 118, 252, 107, 226, 156, 116,
    4, 241, 69, 157, 112, 89, 100, 113, 135, 32, 134, 91, 207, 101, 230, 45, 168, 2, 27, 96, 37, 173, 174, 176, 185, 246, 28, 70, 97, 105, 52, 64,
    126, 15, 85, 71, 163, 35, 221, 81, 175, 58, 195, 92, 249, 206, 186, 197, 234, 38, 44, 83, 13, 110, 133, 40, 132, 9, 211, 223, 205, 244, 65, 129,
    77, 82, 106, 220, 55, 200, 108, 193, 171, 250, 36, 225, 123, 8, 12, 189, 177, 74, 120, 136, 149, 139, 227, 99, 232, 109, 233, 203, 213, 254, 59,
    0, 29, 57, 242, 239, 183, 14, 102, 88, 208, 228, 166, 119, 114, 248, 235, 117, 75, 10, 49, 68, 80, 180, 143, 237, 31, 26, 219, 153, 141, 51, 159,
    17, 131, 20,
];

fn process_check_sum(c: &mut [u8; 16], m: &[u8; 16]) {
    let mut l = c[15] as usize;
    for i in 0..16 {
        c[i] ^= S[(m[i] as usize ^ l) & 0xFF];
        l = c[i] as usize;
    }
}

fn process_block(x: &mut [u8; 48], m: &[u8; 16]) {
    for i in 0..16 {
        x[i + 16] = m[i];
        x[i + 32] = m[i] ^ x[i];
    }
    let mut t = 0;
    for j in 0..18 {
        for k in 0..48 {
            x[k] ^= S[t];
            t = x[k] as usize;
            t = t & 0xFF;
        }
        t = (t + j) % 256;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::digests::test_digest::test_digest;

    #[test]
    fn test() {
        let messages = vec![
            "",
            "a",
            "abc",
            "message digest",
            "abcdefghijklmnopqrstuvwxyz",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            "12345678901234567890123456789012345678901234567890123456789012345678901234567890",
        ];

        let digests = vec![
            "8350e5a3e24c153df2275c9f80692773",
            "32ec01ec4a6dac72c0ab96fb34c0b5d1",
            "da853b0d3f88d99b30283a69e6ded6bb",
            "ab4f496bfb2a530b219ff33031fe06b0",
            "4e8ddff3650292ab5a4108c3aa47940b",
            "da33def2a42df13975352846c30338cd",
            "d5976f79d83d3a0dc9806c3c66f3efd8",
        ];

        test_digest(Md2Digest::new(), &messages, &digests);
    }
}
