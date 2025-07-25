//! BigInteger utilities.
use rand::RngCore;
use crate::math::big_integer::ZERO;
use crate::math::BigInteger;
use crate::{BcError, Result};
use crate::math::raw::{internal_mod, nat};

const MAX_ITERATIONS: usize = 1000;

/// Return a random BigInteger not less than 'min' and not greater than 'max'
///
/// # Arguments
///
/// * `min` - the least value that may be generated
/// * `max` - the greatest value that may be generated
/// * `random` - random the source of randomness
///
/// # Returns
/// A random BigInteger not less than 'min' and not greater than 'max'
///
/// # Panics
/// `min` may not be greater than `max`
pub fn create_random_in_range<TRngCore: RngCore>(min: &BigInteger, max: &BigInteger, random: &mut TRngCore) -> BigInteger {
    let cmp = min.partial_cmp(max);
    if cmp == Some(std::cmp::Ordering::Greater) {
        panic!("'min' may not be greater than 'max'");
    } else if cmp == Some(std::cmp::Ordering::Equal) {
        return min.clone();
    }

    if min.bit_length() > max.bit_length() / 2 {
        return create_random_in_range(&(*ZERO), &max.subtract(min), random);
    }

    for _ in 0..MAX_ITERATIONS {
        let x = BigInteger::with_random(max.bit_length(), random);
        if &x >= min && &x <= max {
            return x;
        }
    }

    BigInteger::with_random(max.subtract(min).bit_length() - 1, random).add(min)
}
pub fn mod_odd_inverse(m: &BigInteger, x: &BigInteger) -> Result<BigInteger> {
    if !m.test_bit(0) {
        return Err(BcError::with_invalid_argument("modulus must be odd"));
    }

    if m.sign() != 1 {
        return Err(BcError::with_invalid_argument("modulus must be positive"));
    }

    let mut x = x.clone();
    if x.sign() < 0 || x.bit_length() > m.bit_length() {
        x = x.modulus(m)?;
    }
    
    let bits = m.bit_length();
    let m = nat::from_big_integer(bits, m)?;
    let x = nat::from_big_integer(bits, &x)?;
    let len = m.len();
    let mut z = nat::create_u32(len);
    if internal_mod::mod_odd_inverse(&m, &x, &mut z) == 0 {
        return Err(BcError::with_arithmetic_error("BigInteger not invertible"));
    }
    Ok(nat::to_big_integer(len, &z)?)
}
pub fn to_unsigned_bytes(length: usize, n : &BigInteger) -> Result<Vec<u8>> {
    let bytes_length = n.get_length_of_u32_vec_unsigned();
    if bytes_length > length {
        return Err(BcError::with_invalid_argument("standard length exceeded"));
    }

    let mut bytes = vec![0u8; length];
    n.copy_to_u8_vec_unsigned_big_endian(&mut bytes[(length - bytes_length)..])?;
    Ok(bytes)
}
pub fn to_unsigned_bytes_inplace(n: &BigInteger, buf: &mut [u8]) -> Result<()> {
    let bytes_length = n.get_length_of_u32_vec_unsigned();
    let buf_length = buf.len();
    if bytes_length > buf_length {
        return Err(BcError::with_invalid_argument("standard length exceeded"));
    }
    buf[..(buf_length - bytes_length)].fill(0);
    n.copy_to_u8_vec_unsigned_big_endian(&mut buf[0..bytes_length])?;
    Ok(())
}
pub fn create_random_big_integer<TRngCore: RngCore>(bit_length: usize, random: &mut TRngCore) -> BigInteger {
    BigInteger::with_random(bit_length, random)
}
// todo