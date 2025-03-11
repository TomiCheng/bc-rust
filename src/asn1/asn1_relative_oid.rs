use std::io::Write;

use crate::math::BigInteger;
use crate::Result;

pub(crate) fn is_valid_identifier(s: &str) -> bool {
    let mut digit_count = 0;

    let mut ch_next: Option<char> = None;
    for ch in s.chars().rev() {
        if ch == '.' {
            if digit_count == 0 || (digit_count > 1 && ch_next == Some('0')) {
                return false;
            }
            digit_count = 0;
        } else if '0' <= ch && ch <= '9' {
            digit_count += 1;
        } else {
            return false;
        }
        ch_next = Some(ch);
    }
    if digit_count == 0 || (digit_count > 1 && ch_next == Some('0')) {
        return false;
    }
    true
}

pub(crate) fn write_field_with_i64(writer: &mut dyn Write, mut value: i64) -> Result<()> {
    let mut result = [0u8; 9];
    let mut pos = 8;
    result[pos] = (value & 0x7F) as u8;
    while value >= (1 << 7) {
        value >>= 7;
        result[{
            pos -= 1;
            pos
        }] = (value & 0x80) as u8;
    }
    writer.write(&result[pos..])?;
    Ok(())
}

pub(crate) fn write_field_with_big_integer(
    writer: &mut dyn Write,
    value: &BigInteger,
) -> Result<()> {
    let byte_count = (value.get_bit_length() + 6) / 7;
    if byte_count == 0 {
        writer.write(&[0])?;
    } else {
        let mut tmp_value = value.clone();
        let mut tmp = vec![0u8; byte_count];
        for i in (0..byte_count).rev() {
            tmp[i] = (tmp_value.get_i32_value() | 0x80) as u8;
            tmp_value = tmp_value.shift_right(7);
        }
        tmp[byte_count - 1] &= 0x7F;
        writer.write(&tmp)?;
    }
    Ok(())
}
