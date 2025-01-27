use std::random::{DefaultRandomSource, RandomSource};

use bc_rust::crypto::Digest;
use bc_rust::math::big_integer::TWO;
use bc_rust::math::primes::{
    enhanced_mr_probable_prime_test, generate_st_random_prime, has_any_small_factors,
    is_mr_probable_prime, is_mr_probable_prime_to_base, SMALL_FACTOR_LIMIT,
};
use bc_rust::math::BigInteger;
use bc_rust::util::big_integers::create_random_in_range;
use bc_rust::{BcError, Result};

const ITERATIONS: u32 = 10;
const PRIME_BITS: usize = 256;
const PRIME_CERTAINTY: u32 = 100;

fn random_prime() -> BigInteger {
    let mut random = DefaultRandomSource::default();
    BigInteger::with_random_certainty(PRIME_BITS, PRIME_CERTAINTY as i32, &mut random).unwrap()
}

fn reference_is_mr_probable_prime(x: &BigInteger, num_bases: u32) -> bool {
    let x_sub_two = x.subtract(&(*TWO));
    let mut random = DefaultRandomSource::default();
    for _ in 0..num_bases {
        let b = create_random_in_range(&(*TWO), &x_sub_two, &mut random);
        if !is_mr_probable_prime_to_base(&x, &b).unwrap() {
            return false;
        }
    }
    true
}

fn is_prime(x: &BigInteger) -> Result<bool> {
    Ok(x.is_probable_prime(PRIME_CERTAINTY as i32)?)
}

#[test]
fn test_has_any_small_factors() {
    for _ in 0..ITERATIONS {
        let prime = random_prime();
        assert!(!has_any_small_factors(&prime).unwrap());

        for small_factor in 2..SMALL_FACTOR_LIMIT {
            let non_prime_with_small_factor = BigInteger::with_i32(small_factor).multiply(&prime);
            assert!(has_any_small_factors(&non_prime_with_small_factor).unwrap());
        }
    }
}

#[test]
fn test_enhanced_mr_probable_prime() {
    let mut random = DefaultRandomSource::default();
    let mr_interations = (PRIME_CERTAINTY + 1) / 2;
    for iterations in 0..ITERATIONS {
        let prime = random_prime();
        let mr = enhanced_mr_probable_prime_test(&prime, &mut random, mr_interations).unwrap();
        assert!(!mr.is_provably_composite());
        assert!(!mr.is_not_prime_power());
        assert_eq!(mr.get_factor(), None);

        let mut prime_power = prime.clone();
        for _ in 0..=(iterations % 8) {
            prime_power = prime_power.multiply(&prime);
        }

        let mr2 =
            enhanced_mr_probable_prime_test(&prime_power, &mut random, mr_interations).unwrap();
        assert!(mr2.is_provably_composite());
        assert!(!mr2.is_not_prime_power());
        assert_eq!(mr2.get_factor(), Some(&prime));

        let non_prime_power = random_prime().multiply(&prime);
        let mr3 =
            enhanced_mr_probable_prime_test(&non_prime_power, &mut random, mr_interations).unwrap();
        assert!(mr3.is_provably_composite());
        assert!(mr3.is_not_prime_power());
        assert_eq!(mr3.get_factor(), None);
    }
}

#[test]
fn test_mr_probable_prime() {
    let mr_iterations = (PRIME_CERTAINTY + 1) / 2;
    let mut random = DefaultRandomSource::default();
    for _ in 0..ITERATIONS {
        let prime = random_prime();
        assert!(is_mr_probable_prime(&prime, &mut random, mr_iterations).unwrap());

        let non_prime = random_prime().multiply(&prime);
        assert!(!is_mr_probable_prime(&non_prime, &mut random, mr_iterations).unwrap());
    }
}

#[test]
fn test_mr_probable_prime_to_base() {
    let mr_iterations = (PRIME_CERTAINTY + 1) / 2;
    for _ in 0..ITERATIONS {
        let prime = random_prime();
        assert!(reference_is_mr_probable_prime(&prime, mr_iterations));

        let non_prime = random_prime().multiply(&prime);
        assert!(!reference_is_mr_probable_prime(&non_prime, mr_iterations));
    }
}

#[test]
fn test_st_random_prime() {
    let mut random = DefaultRandomSource::default();
    let mut digests: [Box<dyn Digest>; 0] = []; // todo!();
    for digest in digests.iter_mut() {
        let mut coincidence_count = 0;
        let mut iterations = 0;
        while iterations < ITERATIONS {
            iterations += 1;
            let mut input_seed = [0u8; 16];
            random.fill_bytes(&mut input_seed);

            let st = match generate_st_random_prime(digest.as_mut(), PRIME_BITS as u32, &input_seed)
            {
                Ok(st3) => st3,
                Err(e) => match e {
                    BcError::InvalidOperation(msg) => {
                        if msg.starts_with("Too many iterations") {
                            iterations -= 1;
                            continue;
                        }
                        panic!("Unexpected error: {}", msg);
                    }
                    _ => panic!("Unexpected error: {}", e),
                },
            };
            assert!(is_prime(st.get_prime()).unwrap());

            let st2 =
                match generate_st_random_prime(digest.as_mut(), PRIME_BITS as u32, &input_seed) {
                    Ok(st3) => st3,
                    Err(e) => match e {
                        BcError::InvalidOperation(msg) => {
                            if msg.starts_with("Too many iterations") {
                                iterations -= 1;
                                continue;
                            }
                            panic!("Unexpected error: {}", msg);
                        }
                        _ => panic!("Unexpected error: {}", e),
                    },
                };
            assert_eq!(st.get_prime(), st2.get_prime());
            assert_eq!(st.get_prime_gen_counter(), st2.get_prime_gen_counter());
            assert_eq!(st.get_prime_seed(), st2.get_prime_seed());

            for i in 0..input_seed.len() {
                input_seed[i] = input_seed[i] ^ 0xFF;
            }

            let st3 =
                match generate_st_random_prime(digest.as_mut(), PRIME_BITS as u32, &input_seed) {
                    Ok(st3) => st3,
                    Err(e) => match e {
                        BcError::InvalidOperation(msg) => {
                            if msg.starts_with("Too many iterations") {
                                iterations -= 1;
                                continue;
                            }
                            panic!("Unexpected error: {}", msg);
                        }
                        _ => panic!("Unexpected error: {}", e),
                    },
                };

            assert_eq!(st.get_prime(), st3.get_prime());
            assert_ne!(st.get_prime_seed(), st3.get_prime_seed());

            if st.get_prime_gen_counter() == st3.get_prime_gen_counter() {
                coincidence_count += 1;
            }
        }

        // todo exception

        assert!(coincidence_count * coincidence_count < ITERATIONS);
    }
}
