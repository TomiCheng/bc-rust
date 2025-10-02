use std::io::Write;
use std::sync::LazyLock;
use crate::BcError;

const ENCODING_TABLE: [u8; 16] = [
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'a', b'b', b'c', b'd', b'e', b'f',
];
const DECODING_TABLE: LazyLock<[u8; 128]> = LazyLock::new(|| {
    let mut table = [0xFFu8; 128];
    for i in 0..ENCODING_TABLE.len() {
        table[ENCODING_TABLE[i] as usize] = i as u8;
    }
    table[b'A' as usize] = table[b'a' as usize];
    table[b'B' as usize] = table[b'b' as usize];
    table[b'C' as usize] = table[b'c' as usize];
    table[b'D' as usize] = table[b'd' as usize];
    table[b'E' as usize] = table[b'e' as usize];
    table[b'F' as usize] = table[b'f' as usize];
    table
});

pub fn decode_to_vec(data: &str) -> Result<Vec<u8>, BcError> {
    let mut result = Vec::with_capacity((data.len() + 1) / 2);
    decode_to_write(data, &mut result)?;
    Ok(result)
}
pub fn decode_to_write(data: &str, out: &mut dyn Write) -> Result<usize, BcError> {
    let mut b1 = 0u8;
    let mut b2 = 0u8;
    let mut length = 0;

    let mut buf = [0u8; 36];
    let mut buf_off = 0;
    let mut end = data.len();

    let mut chars = data.chars().rev();
    while let Some(c) = chars.next() {
        if !ignore(c) {
            break;
        }
        end -= 1;
    }

    let mut chars = data.chars();
    let mut i = 0;
    while i < end {
        while let Some(c) = chars.next() {
            i += 1;
            if ignore(c) {
                continue;
            } else {
                b1 = DECODING_TABLE[c as u8 as usize];
                break;
            }
        }

        while let Some(c) = chars.next() {
            i += 1;
            if ignore(c) {
                continue;
            } else {
                b2 = DECODING_TABLE[c as u8 as usize];
                break;
            }
        }

        if b1 | b2 >= 0x80 {
            return Err(BcError::invalid_data("invalid characters encountered in Hex data"));
        }

        buf[buf_off] = (b1 << 4) | b2;
        buf_off += 1;

        if buf_off == buf.len() {
            out.write(&buf)?;
            buf_off = 0;
        }
        length += 1;
    }

    if buf_off > 0 {
        out.write(&buf[0..buf_off])?;
    }
    Ok(length)
}

fn ignore(c: char) -> bool {
    c == ' ' || c == '\r' || c == '\n' || c == '\t'
}