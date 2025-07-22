use crate::{BcError, Result};
use crate::math::BigInteger;
use crate::util::pack;

const M: u64 = 0xFFFFFFFF;
pub fn add_u32(len: usize, x: &[u32], y: &[u32], z: &mut [u32]) -> u32 {
    let mut c = 0;
    for i in 0..len {
        c += x[i] as u64 + y[i] as u64;
        z[i] = c as u32;
        c >>= 32;
    }
    c as u32
}
pub fn add_33_at(len: usize, x: u32, z: &mut [u32], z_pos: usize) -> u32 {
    debug_assert!(z_pos <= (len - 2));

    let mut c = z[z_pos + 0] as u64 + x as u64;
    z[z_pos + 0] = c as u32;
    c >>= 32;
    c += z[z_pos + 1] as u64 + 1;
    z[z_pos + 1] = c as u32;
    c >>= 32;

    if c == 0 { 0 } else { inc_at(len, z, z_pos + 2) }
}
pub fn add_33_to(len: usize, x: u32, z: &mut [u32]) -> u32 {
    let mut c = z[0] as u64 + x as u64;
    z[0] = c as u32;
    c >>= 32;
    c += z[1] as u64 + 1;
    z[1] = c as u32;
    c >>= 32;
    if c == 0 { 0 } else { inc_at(len, z, 2) }
}
pub fn add_both_to(len: usize, x: &[u32], y: &[u32], z: &mut [u32]) -> u32 {
    let mut c = 0;
    for i in 0..len {
        c += x[i] as u64 + y[i] as u64 + z[i] as u64;
        z[i] = c as u32;
        c >>= 32;
    }
    c as u32
}
pub fn add_dword_at(len: usize, x: u64, z: &mut [u32], z_pos: usize) -> u32 {
    debug_assert!(z_pos <= (len - 2));

    let mut c = z[z_pos + 0] as u64 + x & M;
    z[z_pos + 0] = c as u32;
    c >>= 32;
    c += z[z_pos + 1] as u64 + (x >> 32);
    z[z_pos + 1] = c as u32;
    c >>= 32;

    if c == 0 { 0 } else { inc_at(len, z, z_pos + 2) }
}
pub fn add_dword_to(len: usize, x: u64, z: &mut [u32]) -> u32 {
    let mut c = z[0] as u64 + x & M;
    z[0] = c as u32;
    c >>= 32;
    c += z[1] as u64 + (x >> 32);
    z[1] = c as u32;
    c >>= 32;

    if c == 0 { 0 } else { inc_at(len, z, 2) }
}
pub fn add_to(len: usize, x: &[u32], z: &mut [u32]) -> u32 {
    let mut c = 0;
    for i in 0..len {
        c += x[i] as u64 + z[i] as u64;
        z[i] = c as u32;
        c >>= 32;
    }
    c as u32
}
pub fn add_to_with_in(len: usize, x: &[u32], z: &mut [u32], c_in: u32) -> u32 {
    let mut c = c_in as u64;
    for i in 0..len {
        c += x[i] as u64 + z[i] as u64;
        z[i] = c as u32;
        c >>= 32;
    }
    c as u32
}
pub fn add_to_each_other(len: usize, u: &mut [u32], v: &mut [u32]) -> u32 {
    let mut c = 0;
    for i in 0..len {
        c += u[i] as u64 + v[i] as u64;
        u[i] = c as u32;
        v[i] = c as u32;
        c >>= 32;
    }
    c as u32
}
pub fn add_word_at(len: usize, x: u32, z: &mut [u32], z_pos: usize) -> u32 {
    debug_assert!(z_pos <= (len - 1));

    let mut c = z[z_pos] as u64 + x as u64;
    z[z_pos] = c as u32;
    c >>= 32;

    if c == 0 { 0 } else { inc_at(len, z, z_pos + 1) }
}
pub fn add_word_to(len: usize, x: u32, z: &mut [u32]) -> u32 {
    let mut c = z[0] as u64 + x as u64;
    z[0] = c as u32;
    c >>= 32;
    if c == 0 { 0 } else { inc_at(len, z, 1) }
}
pub fn c_add(len: usize, mask: i32, x: &[u32], y: &[u32], z: &mut [u32]) -> u32 {
    let mask = (!(mask & 1)) as u32;
    let mut c = 0;
    for i in 0..len {
        c += (x[i] & mask) as u64 + (y[i] & mask) as u64;
        z[i] = c as u32;
        c >>= 32;
    }
    c as u32
}
pub fn c_add_to(len: usize, mask: i32, x: &[u32], z: &mut [u32]) -> u32 {
    let mask = (!(mask & 1)) as u32;
    let mut c = 0;
    for i in 0..len {
        c += (z[i] & mask) as u64 + (x[i] & mask) as u64;
        z[i] = c as u32;
        c >>= 32;
    }
    c as u32
}
pub fn c_mov(len: usize, mask: i32, x: &[u32], z: &mut [u32]) {
    let mask = (!(mask & 1)) as u32;
    for i in 0..len {
        let mut z_i = z[i];
        let diff = z_i ^ x[i];
        z_i ^= diff & mask;
        z[i] = z_i;
    }
}
pub fn compare(len: usize, x: &[u32], y: &[u32]) -> i32 {
    for i in (0..len).rev() {
        let x_i = x[i];
        let y_i = y[i];
        if x_i < y_i {
            return -1;
        }
        if x_i > y_i {
            return 1;
        }
    }
    0
}
pub fn copy_u32(len: usize, x: &[u32], z: &mut [u32]) {
    z[..len].copy_from_slice(&x[..len]);
}
pub fn copy_u64(len: usize, x: &[u64], z: &mut [u64]) {
    z[..len].copy_from_slice(&x[..len]);
}
pub fn create_u32(len: usize) -> Vec<u32> {
    vec![0; len]
}
pub fn create_u64(len: usize) -> Vec<u64> {
    vec![0; len]
}
pub fn c_sub(len: usize, mask: i32, x: &[u32], y: &[u32], z: &mut [u32]) -> i32 {
    let mask = (!(mask & 1)) as u32;
    let mut c = 0;
    for i in 0..len {
        c += x[i] as i64 - (y[i] & mask) as i64;
        z[i] = c as u32;
        c >>= 32;
    }
    c as i32
}
pub fn dec(len: usize, z: &mut [u32]) -> i32 {
    for i in 0..len {
        z[i] -= 1;
        if z[i] != u32::MAX {
            return 0;
        }
    }
    -1
}
pub fn dec2(len: usize, x: &[u32], z: &mut [u32]) -> i32 {
    let mut i = 0;
    while i < len {
        let c = x[i] - 1;
        z[i] = c;
        i += 1;
        if c != u32::MAX {
            while i < len {
                z[i] = x[i];
                i += 1;
            }
            return 0;
        }
    }
    -1
}
pub fn dec_at(len: usize, z: &mut [u32], z_pos: usize) -> i32 {
    debug_assert!(z_pos <= len);
    for i in z_pos..len {
        z[i] -= 1;
        if z[i] != u32::MAX {
            return 0;
        }
    }
    -1
}
pub fn eq(len: usize, x: &[u32], y: &mut [u32]) -> bool {
    for i in (0..len).rev() {
        if x[i] != y[i] {
            return false;
        }
    }
    true
}
pub fn equal_to_u32(len: usize, x: &[u32], y: u32) -> u32 {
    let mut d = x[0] ^ y;
    for i in 1..len {
        d |= x[i];
    }
    d = (d >> 1) | (d & 1);
    ((d as i32 - 1) >> 31) as u32
}
pub fn equal_to_u32s(len: usize, x: &[u32], y: &[u32]) -> u32 {
    let mut d = 0;
    for i in 0..len {
        d |= x[i] ^ y[i];
    }
    d = (d >> 1) | (d & 1);
    ((d as i32 - 1) >> 31) as u32
}
// TODO
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
pub fn inc_at(len: usize, z: &mut [u32], z_pos: usize) -> u32 {
    debug_assert!(z_pos <= (len - 1));
    for i in z_pos..len {
        z[i] += 1;
        if z[i] != u32::MIN {
            return 0;
        }
    }
    1
}