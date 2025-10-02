//! Utility methods for generating primes and testing for primality.
//!

use rand::RngCore;
use crate::BcError;
use crate::crypto::Digest;
use crate::crypto::util::pack::Pack;
use crate::math::BigInteger;
use crate::math::big_integer::{ONE, THREE, TWO};
use crate::util::big_integers::with_range_rng;

/// FIPS 186-4 C.6 Shawe-Taylor Random_Prime Routine.
pub struct StRandomPrime {
    prime_gen_counter: usize,
    prime_seed: Vec<u8>,
    prime: BigInteger,
}
impl StRandomPrime {
    fn new(prime: BigInteger, prime_seed: Vec<u8>, prime_gen_counter: usize) -> Self {
        StRandomPrime {
            prime_gen_counter,
            prime_seed,
            prime,
        }
    }
    /// FIPS 186-4 C.6 Shawe-Taylor Random_Prime Routine.
    /// Construct a provable prime number using a hash function.
    ///
    /// # Parameters
    ///
    /// * `hash` - The [`Digest`] instance to use (as "Hash()").
    /// * `length` - The length (in bits) of the prime to be generated. Must be at least 2.
    /// * `input_seed` - The seed to be used for the generation of the requested prime. Cannot be empty.
    ///
    /// # Returns
    /// an [`StRandomPrime`] instance containing the requested prime.
    ///
    pub fn generate(
        hash: &mut dyn Digest,
        length: usize,
        input_seed: &[u8],
    ) -> Result<Self, BcError> {
        if length < 2 || length > 65536 {
            return Err(BcError::invalid_argument(
                "length must be between 2 and 65536",
            ));
        }
        if input_seed.is_empty() {
            return Err(BcError::invalid_argument("input_seed cannot be empty"));
        }
        Self::impl_st_random_prime(hash, length, input_seed.to_vec())
    }
    pub fn prime(&self) -> &BigInteger {
        &self.prime
    }
    pub fn prime_seed(&self) -> &[u8] {
        &self.prime_seed
    }
    pub fn prime_gen_counter(&self) -> usize {
        self.prime_gen_counter
    }

    fn impl_st_random_prime(
        d: &mut dyn Digest,
        length: usize,
        mut prime_seed: Vec<u8>,
    ) -> Result<Self, BcError> {
        let d_len = d.digest_size();
        let c_len = d_len.max(4);

        if length < 33 {
            let mut prime_gen_counter = 0usize;
            let mut c0 = vec![0u8; c_len];
            let mut c1 = vec![0u8; c_len];

            loop {
                hash(d, &mut prime_seed, &mut c0[(c_len - d_len)..])?;
                inc(&mut prime_seed, 1);

                hash(d, &mut prime_seed, &mut c1[(c_len - d_len)..])?;
                inc(&mut prime_seed, 1);

                let mut c =
                    u32::from_be_slice(&c0[(c_len - 4)..]) ^ u32::from_be_slice(&c1[(c_len - 4)..]);
                c &= u32::MAX >> (32 - length);
                c |= (1 << (length - 1)) | 1;

                prime_gen_counter += 1;

                if is_prime32(c) {
                    return Ok(Self::new(
                        BigInteger::from_u32(c),
                        prime_seed,
                        prime_gen_counter,
                    ));
                }

                if prime_gen_counter > (4 * length) {
                    return Err(BcError::invalid_operation(
                        "Too many iterations in generation of prime number",
                    ));
                }
            }
        }
        let rec = Self::impl_st_random_prime(d, (length + 3) / 2, prime_seed)?;
        {
            let c0 = rec.prime;
            prime_seed = rec.prime_seed;
            let prime_gen_counter = rec.prime_gen_counter;

            let out_len = 8 * d_len;
            let iterations = (length - 1) / out_len;

            let old_counter = prime_gen_counter;

            let mut x = hash_gen(d, &mut prime_seed, iterations + 1)?;
            x = x.modulus(&(*ONE).shift_left((length - 1) as isize).set_bit(length - 1))?;

            let c0x2 = c0.shift_left(1);
            let mut tx2 = x.subtract(&(*ONE)).divide(&c0x2)?.add(&(*ONE)).shift_left(1);
            let mut dt = 0;

            let mut c = tx2.multiply(&c0).add(&(*ONE));

            // sieving could be used here in place of the 'HasAnySmallFactors' approach.
            loop {
                if impl_has_any_small_factors(&c)? {
                    inc(&mut prime_seed, (iterations + 1) as i32);
                } else {
                    let mut a = hash_gen(d, &mut prime_seed, iterations + 1)?;
                    a = a.modulus(&c.subtract(&(*THREE)))?.add(&(*TWO));

                    tx2 = tx2.add(&BigInteger::from_i32(dt));
                    dt = 0;

                    let z: BigInteger = a.modulus_pow(&tx2, &c)?;

                    if (&c.gcd(&z.subtract(&(*ONE)))? == &(*ONE))
                        && (&z.modulus_pow(&c0, &c)? == &(*ONE))
                    {
                        return Ok(Self::new(c, prime_seed, prime_gen_counter));
                    }
                }

                if prime_gen_counter >= ((4 * length) + old_counter) {
                    return Err(BcError::invalid_operation(
                        "Too many iterations in generation of prime number",
                    ));
                }

                dt += 2;
                c = c.add(&c0x2);
            }
        }
    }
}

pub const SMALL_FACTOR_LIMIT: u32 = 211;
/// A fast check for small divisors, up to some implementation-specific limit.
///
/// # Parameters
///
/// * `candidate` - The [`BigInteger`] instance to test for division by small factors.
///
/// # Returns
///
/// `true` if the candidate is found to have any small factors, `false` otherwise.
///
/// # Panics
/// If `candidate` is not greater than 0 or has a bit length of 1 or less.
///
pub fn has_any_small_factors(candidate: &BigInteger) -> Result<bool, BcError> {
    check_candidate(candidate, "candidate");
    impl_has_any_small_factors(candidate)
}
/// FIPS 186-4 C.3.1 Miller-Rabin Probabilistic Primality Test.
/// Run several iterations of the Miller-Rabin algorithm with randomly-chosen bases.
///
/// # Parameters
///
/// * `candidate` - The `BigInteger` instance to test for primality.
/// * `random` - The source of randomness to use to choose bases.
/// * `iterations` - The number of randomly-chosen bases to perform the test for.
///
/// # Returns
///
/// `false` if any witness to composites is found amongst the chosen bases (so `candidate` s definitely NOT prime),
/// or else `true` (indicating primality with some probability dependent on the number of iterations that were performed).
///
pub fn is_mr_probable_prime<TRngCore: RngCore>(
    candidate: &BigInteger,
    random: &mut TRngCore,
    iterations: u32,
) -> bool {
    check_candidate(candidate, "candidate");

    if iterations < 1 {
        panic!("iterations must be at least 1");
    }

    if candidate.bit_length() == 2 {
        return true;
    }

    if !candidate.test_bit(0) {
        return false;
    }

    let w = candidate;
    let w_sub_one = candidate.subtract(&(*ONE));
    let w_sub_two = candidate.subtract(&(*TWO));

    let a = w_sub_one.get_lowest_set_bit();
    let m = w_sub_one.shift_right(a as isize);

    for _ in 0..iterations {
        let b = with_range_rng(&(*&TWO), &w_sub_two, random);
        // TODO
        // if !impl_mr_probable_prime_to_base(w, &w_sub_one, &m, a, &b)? {
        //     return false;
        // }
    }
    true
}

fn check_candidate(n: &BigInteger, name: &str)  {
    if !(n.sign() > 0 && n.bit_length() > 1) {
        panic!("{} must be > 0 and have a bit length > 1", name);
    }
}
fn hash(d: &mut dyn Digest, input: &[u8], output: &mut [u8]) -> Result<(), BcError> {
    d.block_update(input)?;
    d.do_final(output)?;
    Ok(())
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
fn hash_gen(d: &mut dyn Digest, seed: &mut [u8], count: usize) -> Result<BigInteger, BcError> {
    let d_len = d.digest_size();
    let mut pos = count * d_len;
    let mut buf = vec![0u8; pos];
    for _ in 0..count {
        pos -= d_len;
        hash(d, seed, &mut buf[pos..])?;
        inc(seed, 1);
    }
    Ok(BigInteger::from_sign_be_slice(1, &buf))
}
fn impl_has_any_small_factors(x: &BigInteger) -> Result<bool, BcError> {
    // Bundle trial divisors into ~32-bit moduli then use fast tests on the ~32-bit remainders.

    let mut m = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23;
    let mut r = x.modulus(&BigInteger::from_u32(m))?.as_i32();
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
        return Ok(true);
    }

    m = 29 * 31 * 37 * 41 * 43;
    r = x.modulus(&BigInteger::from_u32(m))?.as_i32();
    if (r % 29) == 0 || (r % 31) == 0 || (r % 37) == 0 || (r % 41) == 0 || (r % 43) == 0 {
        return Ok(true);
    }

    m = 47 * 53 * 59 * 61 * 67;
    r = x.modulus(&BigInteger::from_u32(m))?.as_i32();
    if (r % 47) == 0 || (r % 53) == 0 || (r % 59) == 0 || (r % 61) == 0 || (r % 67) == 0 {
        return Ok(true);
    }

    m = 71 * 73 * 79 * 83;
    r = x.modulus(&BigInteger::from_u32(m))?.as_i32();
    if (r % 71) == 0 || (r % 73) == 0 || (r % 79) == 0 || (r % 83) == 0 {
        return Ok(true);
    }

    m = 89 * 97 * 101 * 103;
    r = x.modulus(&BigInteger::from_u32(m))?.as_i32();
    if (r % 89) == 0 || (r % 97) == 0 || (r % 101) == 0 || (r % 103) == 0 {
        return Ok(true);
    }

    m = 107 * 109 * 113 * 127;
    r = x.modulus(&BigInteger::from_u32(m))?.as_i32();
    if (r % 107) == 0 || (r % 109) == 0 || (r % 113) == 0 || (r % 127) == 0 {
        return Ok(true);
    }

    m = 131 * 137 * 139 * 149;
    r = x.modulus(&BigInteger::from_u32(m))?.as_i32();
    if (r % 131) == 0 || (r % 137) == 0 || (r % 139) == 0 || (r % 149) == 0 {
        return Ok(true);
    }

    m = 151 * 157 * 163 * 167;
    r = x.modulus(&BigInteger::from_u32(m))?.as_i32();
    if (r % 151) == 0 || (r % 157) == 0 || (r % 163) == 0 || (r % 167) == 0 {
        return Ok(true);
    }

    m = 173 * 179 * 181 * 191;
    r = x.modulus(&BigInteger::from_u32(m))?.as_i32();
    if (r % 173) == 0 || (r % 179) == 0 || (r % 181) == 0 || (r % 191) == 0 {
        return Ok(true);
    }

    m = 193 * 197 * 199 * 211;
    r = x.modulus(&BigInteger::from_u32(m))?.as_i32();
    if (r % 193) == 0 || (r % 197) == 0 || (r % 199) == 0 || (r % 211) == 0 {
        return Ok(true);
    }
    // NOTE: Unit tests depend on SMALL_FACTOR_LIMIT matching the highest small factor tested here.
    Ok(false)
}
fn impl_mr_probable_prime_to_base(
    w: &BigInteger,
    w_sub_one: &BigInteger,
    m: &BigInteger,
    a: i32,
    b: &BigInteger,
) -> Result<bool, BcError> {
    let mut z = b.modulus_pow(m, w)?;
    if z == *ONE || z == *w_sub_one {
        return Ok(true);
    }

    for _ in 1..a {
        z = z.square().modulus(w)?;

        if &z == w_sub_one {
            return Ok(true);
        }

        if z == *ONE {
            return Ok(false);
        }
    }

    Ok(false)
}

#[cfg(test)]
mod tests {
    use crate::crypto::Digest;
    use crate::crypto::digests::{Sha1Digest, Sha256Digest};
    use crate::math::BigInteger;
    use crate::math::big_integer::PrimeBigInteger;
    use crate::math::primes::*;
    use rand::prelude::*;
    use rand::rng;

    const ITERATIONS: usize = 10;
    const PRIME_BITS: usize = 256;
    const PRIME_CERTAINTY: usize = 100;

    #[test]
    fn test_has_any_small_factors() {
        for _ in 0..ITERATIONS {
            let prime = random_prime();
            assert!(!has_any_small_factors(&prime).unwrap());

            for small_factor in 2..SMALL_FACTOR_LIMIT {
                let non_prime_with_small_factor =
                    BigInteger::from_u32(small_factor).multiply(&prime);
                assert!(has_any_small_factors(&non_prime_with_small_factor).unwrap());
            }
        }
    }
    #[test]
    fn test_st_random_prime() {
        let mut random = rng();
        let mut digests: [Box<dyn Digest>; 2] =
            [Box::new(Sha1Digest::new()), Box::new(Sha256Digest::new())];

        digests.iter_mut().for_each(|digest| {
            let mut coincidence_count = 0;
            let mut iterations = 0usize;

            let mut gen_prime = |seed: &[u8], iterations: &mut usize| -> StRandomPrime {
                loop {
                    match StRandomPrime::generate(digest.as_mut(), PRIME_BITS, seed) {
                        Ok(v) => return v,
                        Err(e) if e.to_string().starts_with("Too many iterations") => {
                            *iterations -= 1;
                            continue;
                        }
                        Err(e) => panic!("Unexpected error: {}", e),
                    }
                }
            };

            while iterations < ITERATIONS {
                iterations += 1;
                let mut input_seed = [0u8; 16];
                random.fill_bytes(&mut input_seed);

                let st = gen_prime(&input_seed, &mut iterations);
                assert!(is_prime(st.prime()));

                let st2 = gen_prime(&input_seed, &mut iterations);
                assert_eq!(st.prime(), st2.prime());
                assert_eq!(st.prime_gen_counter(), st2.prime_gen_counter());
                assert_eq!(st.prime_seed(), st2.prime_seed());

                for b in input_seed.iter_mut() {
                    *b ^= 0xFF;
                }

                let st3 = gen_prime(&input_seed, &mut iterations);
                assert_ne!(st.prime(), st3.prime());
                assert_ne!(st.prime_seed(), st3.prime_seed());

                if st.prime_gen_counter() == st3.prime_gen_counter() {
                    coincidence_count += 1;
                }
            }
            assert!(coincidence_count * coincidence_count < ITERATIONS);
        });
    }
    #[test]
    fn test_enhanced_mr_probable_prime() {
        let mut random = rng();
        let mr_interations = (PRIME_CERTAINTY + 1) / 2;
        for iterations in 0..ITERATIONS {
            let prime = random_prime();
            //         //         let mr = enhanced_mr_probable_prime_test(&prime, &mut random, mr_interations).unwrap();
            //         //         assert!(!mr.is_provably_composite());
            //         //         assert!(!mr.is_not_prime_power());
            //         //         assert_eq!(mr.get_factor(), None);
            //         //
            //         //         let mut prime_power = prime.clone();
            //         //         for _ in 0..=(iterations % 8) {
            //         //             prime_power = prime_power.multiply(&prime);
            //         //         }
            //         //
            //         //         let mr2 = enhanced_mr_probable_prime_test(&prime_power, &mut random, mr_interations).unwrap();
            //         //         assert!(mr2.is_provably_composite());
            //         //         assert!(!mr2.is_not_prime_power());
            //         //         assert_eq!(mr2.get_factor(), Some(&prime));
            //         //
            //         //         let non_prime_power = random_prime().multiply(&prime);
            //         //         let mr3 =
            //         //             enhanced_mr_probable_prime_test(&non_prime_power, &mut random, mr_interations).unwrap();
            //         //         assert!(mr3.is_provably_composite());
            //         //         assert!(mr3.is_not_prime_power());
            //         //         assert_eq!(mr3.get_factor(), None);
        }
    }
    //
    // #[test]
    // fn test_mr_probable_prime() {
    //     let mr_iterations = (PRIME_CERTAINTY + 1) / 2;
    //     let mut random = rng();
    //     for _ in 0..ITERATIONS {
    //         let prime = random_prime();
    //         assert!(is_mr_probable_prime(&prime, &mut random, mr_iterations).unwrap());
    //
    //         let non_prime = random_prime().multiply(&prime);
    //         assert!(!is_mr_probable_prime(&non_prime, &mut random, mr_iterations).unwrap());
    //     }
    // }
    //
    // #[test]
    // fn test_mr_probable_prime_to_base() {
    //     let mr_iterations = (PRIME_CERTAINTY + 1) / 2;
    //     for _ in 0..ITERATIONS {
    //         let prime = random_prime();
    //         assert!(reference_is_mr_probable_prime(&prime, mr_iterations));
    //
    //         let non_prime = random_prime().multiply(&prime);
    //         assert!(!reference_is_mr_probable_prime(&non_prime, mr_iterations));
    //     }
    // }
    //

    fn random_prime() -> BigInteger {
        let mut random = rng();
        BigInteger::create_probable_prime(PRIME_BITS, PRIME_CERTAINTY, &mut random).unwrap()
    }
    fn is_prime(x: &BigInteger) -> bool {
        x.is_probable_prime(PRIME_CERTAINTY).unwrap()
    }
    // fn reference_is_mr_probable_prime(x: &BigInteger, num_bases: u32) -> bool {
    //     let x_sub_two = x.subtract(&(*TWO));
    //     let mut random = rng();
    //     for _ in 0..num_bases {
    //         let b = create_random_in_range(&(*TWO), &x_sub_two, &mut random);
    //         if !is_mr_probable_prime_to_base(&x, &b).unwrap() {
    //             return false;
    //         }
    //     }
    //     true
    // }
}
