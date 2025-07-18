use rand::RngCore;
use crate::math::big_integer::ZERO;
use crate::math::BigInteger;

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