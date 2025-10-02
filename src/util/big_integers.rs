use std::cmp::Ordering;
use rand::RngCore;
use crate::math::big_integer::{RandomBigInteger, ZERO};
use crate::math::BigInteger;

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
pub fn with_range_rng<TRngCore: RngCore>(min: &BigInteger, max: &BigInteger, random: &mut TRngCore) -> BigInteger {
    let cmp = min.partial_cmp(max);
    if cmp == Some(Ordering::Greater) {
        panic!("'min' may not be greater than 'max'");
    } else if cmp == Some(Ordering::Equal) {
        return min.clone();
    }

    if min.bit_length() > max.bit_length() / 2 {
        return with_range_rng(&(*ZERO), &max.subtract(min), random);
    }

    for _ in 0..MAX_ITERATIONS {
        let x = BigInteger::with_rng(max.bit_length(), random);
        if &x >= min && &x <= max {
            return x;
        }
    }

    BigInteger::with_rng(max.subtract(min).bit_length() - 1, random).add(min)
}

const MAX_ITERATIONS: usize = 1000;