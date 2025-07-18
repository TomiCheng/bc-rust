use crate::{BcError, Result};
use crate::math::BigInteger;
use crate::util::pack;

pub fn get_bit_length(len: usize, x: &[u32]) -> usize {
    for i in (0..len).rev() {
        let x_i = x[i];
        if x_i != 0 {
            return (i * 32) + 32 - x_i.leading_zeros() as usize;
        }
    }
    0
}

// TODO

pub(crate) fn gte(len: usize, x: &[u32], y: &[u32]) -> bool {
    for i in (0..len).rev() {
        let x_i = x[i];
        let y_i = y[i];
        if x_i < y_i {
            return false;
        }
        if x_i > y_i {
            return true;
        }
    }
    true
}

// TODO

pub(crate) fn less_than(len: usize, x: &[u32], y: &[u32]) -> i32 {
    let mut c = 0;
    for i in 0..len {
        c += x[i] as i64 - y[i] as i64;
        c >>= 32;
    }

    debug_assert!(c == 0 || c == -1);
    c as i32
}

pub(crate) fn from_big_integer(bits: usize, x: &BigInteger) -> Result<Vec<u32>> {
    if x.sign() < 0 || x.bit_length() > bits {
        return Err(BcError::with_invalid_argument(""));
    }

    let len = get_length_for_bits(bits)?;
    let mut z = create_u32(len);

    let x_len = x.get_length_of_u32_vec_unsigned();
    x.copy_to_u32_vec_unsigned_little_endian(&mut z[..x_len])?;
    z[x_len..].fill(0x00);
    Ok(z)
}
pub(crate) fn get_length_for_bits(bits: usize) -> Result<usize> {
    if bits < 1 {
        return Err(BcError::with_invalid_argument(""));
    }

    Ok((bits + 31) >> 5)
}
pub(crate) fn create_u32(len: usize) -> Vec<u32> {
    vec![0; len]
}
pub(crate) fn to_big_integer(len: usize, x: &[u32]) -> Result<BigInteger> {
    if cfg!(target_endian = "little") {
        unsafe {
            let bs = std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * size_of::<u32>());
            BigInteger::with_sign_buffer_big_endian(1, &bs, false)
        }
    } else {
        let bs_len = len << 2;
        let mut bs = vec![0u8; bs_len];
        pack::u32_to_u8_vec_le(x, &mut bs);
        BigInteger::with_sign_buffer_big_endian(1, &bs, false)
    }
}