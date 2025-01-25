//! Utility methods for generating primes and testing for primality.
//!

use std::random::RandomSource;

use super::big_integer::BigInteger;
use crate::{
    crypto::Digest,
    math::big_integer::{ONE, THREE, TWO},
    util::pack::be_to_u32,
};

/// FIPS 186-4 C.6 Shawe-Taylor Random_Prime Routine.  
/// Construct a provable prime number using a hash function.
/// 
/// # Parameters
/// 
/// * `hash` - The `Digest` instance to use (as "Hash()").
/// * `length` - The length (in bits) of the prime to be generated. Must be at least 2.
/// * `input_seed` - The seed to be used for the generation of the requested prime. Cannot be null or empty.
/// 
/// # Returns
/// an `StOutput` instance containing the requested prime.
/// 
/// # Panics
/// - `length` must be at least 2
/// - `input_seed` cannot be empty
pub fn generate_st_random_prime(hash: &mut dyn Digest, length: u32, input_seed: &[u8]) -> StOutput {
    if length < 2 {
        panic!("length must be at least 2");
    }
    if input_seed.len() == 0 {
        panic!("cannot by empty input_seed");
    }

    impl_st_random_prime(hash, length, input_seed.to_vec())
}

pub fn enhanced_mr_probable_prime_test(candidate: &BigInteger, random: &mut dyn RandomSource, iterations: u32) -> MrOutput {
    check_candidate(candidate, "candidate");
    if iterations < 1 {
        panic!("iterations must be at least 1");
    }

    if *candidate.get_bit_length() == 2 {
        return MrOutput::with_probaly_prime();
    }

    if !candidate.test_bit(0) {
        return MrOutput::with_provably_composite_with_factor((*TWO).clone());
    }

    let mut w = candidate;
    let w_sub_one = candidate.subtract(&(*ONE));
    let w_sub_two = candidate.subtract(&(*TWO));

    let a = w_sub_one.get_lowest_set_bit();
    let m = w_sub_one.shift_right(a);

    for i in 0..iterations {
        //let b = B
    }
    todo!();


}

/// Used to return the output from the `EnhancedMRProbablePrimeTest` Enhanced Miller-Rabin Probabilistic Primality Test
pub struct MrOutput {
    provably_composite: bool,
    factor: Option<BigInteger>,
}

impl MrOutput {
    fn new(provably_composite: bool, factor: Option<BigInteger>) -> Self {
        MrOutput {
            provably_composite,
            factor,
        }
    }

    pub fn get_factor(&self) -> Option<&BigInteger> {
        self.factor.as_ref()
    }

    pub(crate) fn with_probaly_prime() -> Self {
        Self::new(false, None)
    }

    pub(crate) fn with_provably_composite_with_factor(factor: BigInteger) -> Self {
        Self::new(true, Some(factor))
    }

    pub(crate) fn with_provably_composite_not_prime_power() -> Self {
        Self::new(true, None)
    }

    pub fn is_provably_composite(&self) -> bool {
        self.provably_composite
    }

    pub fn is_prime_power(&self) -> bool {
        self.provably_composite && self.factor.is_none()
    }
}

pub struct StOutput {
    prime_gen_couter: u32,
    prime_seed: Vec<u8>,
    prime: BigInteger,
}

impl StOutput {
    pub(crate) fn new(prime: BigInteger, prime_seed: Vec<u8>, prime_gen_couter: u32) -> Self {
        StOutput {
            prime_gen_couter,
            prime_seed,
            prime,
        }
    }
}

fn impl_st_random_prime(d: &mut dyn Digest, length: u32, mut prime_seed: Vec<u8>) -> StOutput {
    let d_len = d.get_digest_size();
    let c_len = d_len.max(4);

    if length < 33 {
        let mut prime_gen_counter = 0;
        let mut c0 = vec![0u8; c_len];
        let mut c1 = vec![0u8; c_len];

        loop {
            hash(d, &mut prime_seed, &mut c0[(c_len - d_len)..]);
            inc(&mut prime_seed, 1);

            hash(d, &mut prime_seed, &mut c1[(c_len - d_len)..]);
            inc(&mut prime_seed, 1);

            let mut c = be_to_u32(&c0[(c_len - 4)..]) ^ be_to_u32(&c1[(c_len - 4)..]);
            c &= u32::MAX >> (32 - length);
            c |= (1 << (length - 1)) | 1;

            prime_gen_counter += 1;

            if is_prime32(c) {
                return StOutput::new(
                    BigInteger::with_u32(c),
                    prime_seed,
                    prime_gen_counter as u32,
                );
            }

            if prime_gen_counter > (4 * length) {
                panic!("Too many iterations in generation of prime number");
            }
        }
    }

    let rec = impl_st_random_prime(d, (length + 3) / 2, prime_seed);
    {
        let c0 = rec.prime;
        prime_seed = rec.prime_seed;
        let prime_gen_counter = rec.prime_gen_couter;

        let out_len = 8 * d_len;
        let iterations = (length - 1) / out_len as u32;

        let old_counter = prime_gen_counter;

        let mut x = hash_gen(d, &mut prime_seed, iterations + 1);
        x = x.r#mod(
            &(*ONE)
                .shift_left((length - 1) as i32)
                .set_bit((length - 1) as usize),
        );

        let c0x2 = c0.shift_left(1);
        let mut tx2 = x.subtract(&(*ONE)).divide(&c0x2).add(&(*ONE)).shift_left(1);
        let mut dt = 0;

        let mut c = tx2.multiply(&c0).add(&(*ONE));

        // TODO Since the candidate primes are generated by constant steps ('c0x2'),
        // sieving could be used here in place of the 'HasAnySmallFactors' approach.

        loop {
            if impl_has_any_small_factors(&c) {
                inc(&mut prime_seed, (iterations + 1) as i32);
            } else {
                let mut a = hash_gen(d, &mut prime_seed, iterations + 1);
                a = a.r#mod(&c.subtract(&(*THREE))).add(&(*TWO));

                tx2 = tx2.add(&BigInteger::with_i32(dt));
                dt = 0;

                let z: BigInteger = a.mod_pow(&tx2, &c);

                if (&c.gcd(&z.subtract(&(*ONE))) == &(*ONE)) && (&z.mod_pow(&c0, &c) == &(*ONE)) {
                    return StOutput::new(c, prime_seed, prime_gen_counter);
                }

                if prime_gen_counter >= ((4 * length) + old_counter) {
                    panic!("Too many iterations in generation of prime number");
                }

                dt += 2;
                c = c.add(&c0x2);
            }
        }
    }
}

fn hash(d: &mut dyn Digest, input: &[u8], output: &mut [u8]) {
    d.block_update(input);
    d.do_final(output);
}

fn inc(seed: &mut [u8], mut c: i32) {
    let mut pos = seed.len() as isize;
    while c > 0 && {
        pos -= 1;
        pos
    } >= 0
    {
        c += seed[pos as usize] as i32;
        seed[pos as usize] = c as u8;
        c >>= 8;
    }
}

fn is_prime32(x: u32) -> bool {
    // Use wheel factorization with 2, 3, 5 to select trial divisors.
    if x < 32 {
        return ((1 << x as i32) & 0b0010_0000_1000_1010_0010_1000_1010_1100) != 0;
    }
    if ((1 << (x % 30) as i32) & 0b1010_0000_1000_1010_0010_1000_1000_0010u32) == 0 {
        return false;
    }

    let ds = [1u32, 7, 11, 13, 17, 19, 23, 29];
    let mut b = 0;
    let mut pos = 1;
    loop {
        // Trial division by wheel-selected divisors
        while pos < ds.len() {
            let d = b + ds[pos];
            if x % d == 0 {
                return false;
            }
            pos += 1;
        }

        b += 30;

        if (b >> 16 != 0) || (b * b >= x) {
            return true;
        }

        pos = 0;
    }
}

fn hash_gen(d: &mut dyn Digest, seed: &mut [u8], count: u32) -> BigInteger {
    let d_len = d.get_digest_size();
    let mut pos = count as usize * d_len;
    let mut buf = vec![0u8; pos];
    for _ in 0..count {
        pos -= d_len;
        hash(d, seed, &mut buf[pos..]);
        inc(seed, 1);
    }
    BigInteger::with_sign_buffer(1, &buf).expect("invalid sing")
}

fn impl_has_any_small_factors(x: &BigInteger) -> bool {
    // Bundle trial divisors into ~32-bit moduli then use fast tests on the ~32-bit remainders.

    let mut m = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23;
    let mut r = x.r#mod(&BigInteger::with_u32(m)).get_i32_value();
    if (r % 2) == 0
        || (r % 3) == 0
        || (r % 5) == 0
        || (r % 7) == 0
        || (r % 11) == 0
        || (r % 13) == 0
        || (r % 17) == 0
        || (r % 19) == 0
        || (r % 23) == 0
    {
        return true;
    }

    m = 29 * 31 * 37 * 41 * 43;
    r = x.r#mod(&BigInteger::with_u32(m)).get_i32_value();
    if (r % 29) == 0
        || (r % 31) == 0
        || (r % 37) == 0
        || (r % 41) == 0
        || (r % 43) == 0
    {
        return true;
    }

    m = 47 * 53 * 59 * 61 * 67;
    r = x.r#mod(&BigInteger::with_u32(m)).get_i32_value();
    if (r % 47) == 0
        || (r % 53) == 0
        || (r % 59) == 0
        || (r % 61) == 0
        || (r % 67) == 0
    {
        return true;
    }

    m = 71 * 73 * 79 * 83;
    r = x.r#mod(&BigInteger::with_u32(m)).get_i32_value();
    if (r % 71) == 0 || (r % 73) == 0 || (r % 79) == 0 || (r % 83) == 0 {
        return true;
    }

    m = 89 * 97 * 101 * 103;
    r = x.r#mod(&BigInteger::with_u32(m)).get_i32_value();
    if (r % 89) == 0 || (r % 97) == 0 || (r % 101) == 0 || (r % 103) == 0 {
        return true;
    }

    m = 107 * 109 * 113 * 127;
    r = x.r#mod(&BigInteger::with_u32(m)).get_i32_value();
    if (r % 107) == 0 || (r % 109) == 0 || (r % 113) == 0 || (r % 127) == 0 {
        return true;
    }

    m = 131 * 137 * 139 * 149;
    r = x.r#mod(&BigInteger::with_u32(m)).get_i32_value();
    if (r % 131) == 0 || (r % 137) == 0 || (r % 139) == 0 || (r % 149) == 0 {
        return true;
    }

    m = 151 * 157 * 163 * 167;
    r = x.r#mod(&BigInteger::with_u32(m)).get_i32_value();
    if (r % 151) == 0 || (r % 157) == 0 || (r % 163) == 0 || (r % 167) == 0 {
        return true;
    }

    m = 173 * 179 * 181 * 191;
    r = x.r#mod(&BigInteger::with_u32(m)).get_i32_value();
    if (r % 173) == 0 || (r % 179) == 0 || (r % 181) == 0 || (r % 191) == 0 {
        return true;
    }

    m = 193 * 197 * 199 * 211;
    r = x.r#mod(&BigInteger::with_u32(m)).get_i32_value();
    if (r % 193) == 0 || (r % 197) == 0 || (r % 199) == 0 || (r % 211) == 0 {
        return true;
    }
    // NOTE: Unit tests depend on SMALL_FACTOR_LIMIT matching the highest small factor tested here.
    return false;
}

fn check_candidate(n: &BigInteger, name: &str) {
    if n.get_sign_value() < 1 || *n.get_bit_length() < 2 {
        panic!("{} must be non-null and >= 2", name);
    }
}