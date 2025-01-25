use bc_rust::math::big_integer::{ONE, THREE, TWO, ZERO};
use bc_rust::math::BigInteger;
use std::random::DefaultRandomSource;
use std::sync::LazyLock;

static MINUS_ONE: LazyLock<BigInteger> = LazyLock::new(|| BigInteger::with_i32(-1));
static MINUS_TWO: LazyLock<BigInteger> = LazyLock::new(|| BigInteger::with_i32(-2));
const FIRST_PRIMES: [i32; 10] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
const NON_PRIMES: [i32; 10] = [0, 1, 4, 10, 20, 21, 22, 25, 26, 27];
const MERSENNE_PRIME_EXPONENTS: [i32; 10] = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89];
const NON_PRIME_EXPONENTS: [i32; 10] = [1, 4, 6, 9, 11, 15, 23, 29, 37, 41];

#[test]
fn mono_bug_81857() {
    let b = BigInteger::with_string("18446744073709551616").expect("error");
    //let exp = (*TWO).clone();
    let mod_ = BigInteger::with_string("48112959837082048697").expect("error");
    let expected = BigInteger::with_string("4970597831480284165").expect("error");

    let manual = b.multiply(&b).r#mod(&mod_);
    assert_eq!(expected, manual, "b * b % mod");
}

#[test]
fn test_abs() {
    assert_eq!((*ZERO).clone(), (*ZERO).clone().abs());
    assert_eq!((*ONE).clone(), (*ONE).clone().abs());
    assert_eq!((*ONE).clone(), (*MINUS_ONE).clone().abs());
    assert_eq!((*TWO).clone(), (*TWO).clone().abs());
    assert_eq!((*TWO).clone(), (*MINUS_TWO).clone().abs());
}

#[test]
fn test_add() {
    for i in -10..=10 {
        for j in -10..=10 {
            let a = BigInteger::with_i32(i + j);
            let b = BigInteger::with_i32(i);
            let c = BigInteger::with_i32(j);
            let d = b.add(&c);
            assert_eq!(a, d, "Problem {} ADD {} should be {}", i, j, i + j);
        }
    }
}

#[test]
fn test_and() {
    for i in -10..=10 {
        for j in -10..=10 {
            let a = BigInteger::with_i32(i & j);
            let b = BigInteger::with_i32(i);
            let c = BigInteger::with_i32(j);
            let d = b.and(&c);
            assert_eq!(a, d, "Problem {} AND {} should be {}", i, j, i & j);
        }
    }
}

#[test]
fn test_and_not() {
    for i in -10..=10 {
        for j in 1..=10 {
            let a = BigInteger::with_i32(i & !j);
            let b = BigInteger::with_i32(i);
            let c = BigInteger::with_i32(j);
            let d = b.and_not(&c);
            assert_eq!(a, d, "Problem {} AND NOT {} should be {}", i, j, i & !j);
        }
    }
}

#[test]
fn test_bit_count() {
    assert_eq!(0, *(*ZERO).get_bit_count());
    assert_eq!(1, *(*ONE).get_bit_count());
    assert_eq!(0, *(*MINUS_ONE).get_bit_count());
    assert_eq!(1, *(*TWO).get_bit_count());
    assert_eq!(1, *(*MINUS_TWO).get_bit_count());
    for i in 0..100 {
        let pow2 = (*ONE).shift_left(i);
        assert_eq!(1, *pow2.get_bit_count());
        assert_eq!(i as usize, *pow2.negate().get_bit_count(), "{}", i);
    }

    let mut random = DefaultRandomSource::default();

    for _ in 0..10 {
        let test = BigInteger::with_random_certainty(128, 0, &mut random);
        let mut bit_count = 0usize;

        //println!("bit length: {}, bit count: {}", *test.get_bit_length(), *test.get_bit_count());

        for bit in 0..*test.get_bit_length() {
            if test.test_bit(bit) {
                bit_count += 1;
            }
        }

        assert_eq!(bit_count, *test.get_bit_count());
    }
}

#[test]
fn test_bit_length() {
    assert_eq!(0, *(*ZERO).get_bit_length());
    assert_eq!(1, *(*ONE).get_bit_length());
    assert_eq!(0, *(*MINUS_ONE).get_bit_length());
    assert_eq!(2, *(*TWO).get_bit_length());
    assert_eq!(1, *(*MINUS_TWO).get_bit_length());

    let mut random = DefaultRandomSource::default();

    for i in 0..100 {
        let bit = i + (std::random::random::<usize>() % 64);
        //println!("{}", bit);
        let odd = BigInteger::with_random(bit, &mut random)
            .set_bit(bit + 1)
            .set_bit(0);
        let pow2 = (*ONE).shift_left(bit as i32);

        assert_eq!(bit + 2, *odd.get_bit_length(), "t1");
        assert_eq!(bit + 2, *odd.negate().get_bit_length(), "t2");
        assert_eq!(bit + 1, *pow2.get_bit_length(), "t3");
        assert_eq!(bit, *pow2.negate().get_bit_length(), "t4");
    }
}

#[test]
fn test_clear_bit() {
    assert_eq!(&(*ZERO), &(*ZERO).clear_bit(0));
    assert_eq!(&(*ZERO), &(*ONE).clear_bit(0));
    assert_eq!(&(*TWO), &(*TWO).clear_bit(0));

    assert_eq!(&(*ZERO), &(*ZERO).clear_bit(1));
    assert_eq!(&(*ONE), &(*ONE).clear_bit(1));
    assert_eq!(&(*ZERO), &(*TWO).clear_bit(1));

    // TODO Tests for clearing bits in negative numbers

    // TODO Tests for clearing extended bits
    let mut random = DefaultRandomSource::default();
    for _ in 0..10 {
        let n = BigInteger::with_random(128, &mut random);

        for _ in 0..10 {
            let pos = std::random::random::<usize>() % 128;
            let m = n.clear_bit(pos);
            assert_ne!(m.shift_right(pos as i32).remainder(&(*TWO)), *ONE);
        }
    }

    for i in 0..100 {
        let pow2 = (*ONE).shift_left(i);
        let minus_pow2 = pow2.negate();

        assert_eq!((*ZERO), pow2.clear_bit(i as usize));

        let right = minus_pow2.clear_bit(i as usize);
        assert_eq!(
            minus_pow2.shift_left(1),
            right,
            "i = {}, minus_pow2 = {:?}",
            i,
            minus_pow2
        );

        let big_i = BigInteger::with_i32(i as i32);
        let neg_i = big_i.negate();

        for j in 0..10 {
            assert_eq!(
                big_i.and_not(&(*ONE).shift_left(j)),
                big_i.clear_bit(j as usize),
                "i = {}, j = {}",
                i,
                j
            );
            assert_eq!(
                neg_i.and_not(&(*ONE).shift_left(j)),
                neg_i.clear_bit(j as usize),
                "i = {}, j = {}",
                i,
                j
            );
        }
    }
}

#[test]
fn test_compare_to() {
    assert_eq!(
        Some(std::cmp::Ordering::Equal),
        (*MINUS_TWO).partial_cmp(&(*MINUS_TWO))
    );
    assert_eq!(
        Some(std::cmp::Ordering::Less),
        (*MINUS_TWO).partial_cmp(&(*MINUS_ONE))
    );
    assert_eq!(
        Some(std::cmp::Ordering::Less),
        (*MINUS_TWO).partial_cmp(&(*ZERO))
    );
    assert_eq!(
        Some(std::cmp::Ordering::Less),
        (*MINUS_TWO).partial_cmp(&(*ONE))
    );
    assert_eq!(
        Some(std::cmp::Ordering::Less),
        (*MINUS_TWO).partial_cmp(&(*TWO))
    );

    assert_eq!(
        Some(std::cmp::Ordering::Greater),
        (*MINUS_ONE).partial_cmp(&(*MINUS_TWO))
    );
    assert_eq!(
        Some(std::cmp::Ordering::Equal),
        (*MINUS_ONE).partial_cmp(&(*MINUS_ONE))
    );
    assert_eq!(
        Some(std::cmp::Ordering::Less),
        (*MINUS_ONE).partial_cmp(&(*ZERO))
    );
    assert_eq!(
        Some(std::cmp::Ordering::Less),
        (*MINUS_ONE).partial_cmp(&(*ONE))
    );
    assert_eq!(
        Some(std::cmp::Ordering::Less),
        (*MINUS_ONE).partial_cmp(&(*TWO))
    );

    assert_eq!(
        Some(std::cmp::Ordering::Greater),
        (*ZERO).partial_cmp(&(*MINUS_TWO))
    );
    assert_eq!(
        Some(std::cmp::Ordering::Greater),
        (*ZERO).partial_cmp(&(*MINUS_ONE))
    );
    assert_eq!(
        Some(std::cmp::Ordering::Equal),
        (*ZERO).partial_cmp(&(*ZERO))
    );
    assert_eq!(Some(std::cmp::Ordering::Less), (*ZERO).partial_cmp(&(*ONE)));
    assert_eq!(Some(std::cmp::Ordering::Less), (*ZERO).partial_cmp(&(*TWO)));

    assert_eq!(
        Some(std::cmp::Ordering::Greater),
        (*ONE).partial_cmp(&(*MINUS_TWO))
    );
    assert_eq!(
        Some(std::cmp::Ordering::Greater),
        (*ONE).partial_cmp(&(*MINUS_ONE))
    );
    assert_eq!(
        Some(std::cmp::Ordering::Greater),
        (*ONE).partial_cmp(&(*ZERO))
    );
    assert_eq!(Some(std::cmp::Ordering::Equal), (*ONE).partial_cmp(&(*ONE)));
    assert_eq!(Some(std::cmp::Ordering::Less), (*ONE).partial_cmp(&(*TWO)));

    assert_eq!(
        Some(std::cmp::Ordering::Greater),
        (*TWO).partial_cmp(&(*MINUS_TWO))
    );
    assert_eq!(
        Some(std::cmp::Ordering::Greater),
        (*TWO).partial_cmp(&(*MINUS_ONE))
    );
    assert_eq!(
        Some(std::cmp::Ordering::Greater),
        (*TWO).partial_cmp(&(*ZERO))
    );
    assert_eq!(
        Some(std::cmp::Ordering::Greater),
        (*TWO).partial_cmp(&(*ONE))
    );
    assert_eq!(Some(std::cmp::Ordering::Equal), (*TWO).partial_cmp(&(*TWO)));
}

#[test]
fn test_constructors() {
    assert_eq!(&(*ZERO), &BigInteger::with_buffer(&[0u8]).unwrap());
    assert_eq!(&(*ZERO), &BigInteger::with_buffer(&[0u8, 0u8]).unwrap());

    let mut random = DefaultRandomSource::default();
    for i in 0..10 {
        let m = BigInteger::with_random_certainty(i + 3, 0, &mut random).test_bit(0);
        assert!(m, "i = {}", i);
    }

    // TODO Other constructors
}

#[test]
#[should_panic(expected = "divide by zero")]
fn test_divide_01() {
    for i in -5..=5 {
        let m = BigInteger::with_i32(i);
        m.divide(&(*ZERO));
    }
}

#[test]
fn test_divide_02() {
    let product = 1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9;
    let product_plus = product + 1;

    let big_product = BigInteger::with_i32(product);
    let big_product_plus = BigInteger::with_i32(product_plus);

    for divisor in 1..10 {
        // Exact division
        let expected = BigInteger::with_i32(product / divisor);

        assert_eq!(
            expected,
            big_product.divide(&BigInteger::with_i32(divisor)),
            "divisor = {}",
            divisor
        );
        assert_eq!(
            expected.negate(),
            big_product.negate().divide(&BigInteger::with_i32(divisor))
        );
        assert_eq!(
            expected.negate(),
            big_product.divide(&BigInteger::with_i32(divisor).negate())
        );
        assert_eq!(
            expected,
            big_product
                .negate()
                .divide(&BigInteger::with_i32(divisor).negate())
        );

        let expected = BigInteger::with_i32((product + 1) / divisor);

        assert_eq!(
            expected,
            big_product_plus.divide(&BigInteger::with_i32(divisor))
        );
        assert_eq!(
            expected.negate(),
            big_product_plus
                .negate()
                .divide(&BigInteger::with_i32(divisor))
        );
        assert_eq!(
            expected.negate(),
            big_product_plus.divide(&BigInteger::with_i32(divisor).negate())
        );
        assert_eq!(
            expected,
            big_product_plus
                .negate()
                .divide(&BigInteger::with_i32(divisor).negate())
        );
    }
}

#[test]
fn test_divide_03() {
    let mut random = DefaultRandomSource::default();
    for req in 0..10 {
        let a = BigInteger::with_random_certainty(100 - req, 0, &mut random);
        let b = BigInteger::with_random_certainty(100 + req, 0, &mut random);
        let c = BigInteger::with_random_certainty(10 + req, 0, &mut random);
        let d = a.multiply(&b).add(&c);
        let e = d.divide(&a);

        assert_eq!(b, e);
    }

    // Special tests for power of two since uses different code path internally
    for _ in 0..100 {
        let shift = (std::random::random::<usize>() % 64) as i32;
        let a = (*ONE).shift_left(shift);
        let b = BigInteger::with_random(64 + std::random::random::<usize>() % 64, &mut random);
        let b_shift = b.shift_right(shift);

        assert_eq!(b_shift, b.divide(&a),
            "shift = {}, b = {:?}", shift, b.to_string_with_radix(16));
        assert_eq!(b_shift.negate(), b.divide(&a.negate()),
            "shift = {}, b = {:?}", shift, b.to_string_with_radix(16));
        assert_eq!(b_shift.negate(), b.negate().divide(&a),
            "shift = {}, b = {:?}", shift, b.to_string_with_radix(16));
        assert_eq!(b_shift, b.negate().divide(&a.negate()),
            "shift = {}, b = {:?}", shift, b.to_string_with_radix(16));
    }
}

#[test]
fn test_divide_04() {

    // Regression
    let shift = 63;
    let a = (*ONE).shift_left(shift);
    let b = BigInteger::with_i64(0x2504b470dc188499);
    let b_shift = b.shift_right(shift);

    assert_eq!(b_shift, b.divide(&a),
        "shift = {}, b = {:?}", shift, b.to_string_with_radix(16));
    assert_eq!(b_shift.negate(), b.divide(&a.negate()),
        "shift = {}, b = {:?}", shift, b.to_string_with_radix(16));
    assert_eq!(b_shift.negate(), b.negate().divide(&a),
        "shift = {}, b = {:?}", shift, b.to_string_with_radix(16));
    assert_eq!(b_shift, b.negate().divide(&a.negate()),
        "shift = {}, b = {:?}", shift, b.to_string_with_radix(16));
}

#[test]
fn test_divide_and_remainder() {
    // TODO More basic tests
    let mut random = DefaultRandomSource::default();
    let n = BigInteger::with_random(48, &mut random);
    let mut qr = n.divide_and_remainder(&n);
    assert_eq!((*ONE), qr.0);
    assert_eq!((*ZERO), qr.1);

    qr = n.divide_and_remainder(&(*ONE));
    assert_eq!(&n, &qr.0);
    assert_eq!(&(*ZERO), &qr.1);

    for rep in 0..10 {
        let a = BigInteger::with_random_certainty(100 - rep, 0, &mut random);
        let b = BigInteger::with_random_certainty(100 + rep, 0, &mut random);
        let c = BigInteger::with_random_certainty(10 + rep, 0, &mut random);
        let d = a.multiply(&b).add(&c);
        let es = d.divide_and_remainder(&a);

        assert_eq!(&b, &es.0);
        assert_eq!(&c, &es.1);
    }

    // Special tests for power of two since uses different code path internally
    for _ in 0..100 {
        let shift = (std::random::random::<usize>() % 64) as i32;
        let a = (*ONE).shift_left(shift);
        let b = BigInteger::with_random(64 + std::random::random::<usize>() % 64, &mut random);
        let b_shift = b.shift_right(shift);
        let b_mod = b.and(&a.subtract(&(*ONE)));

        qr = b.divide_and_remainder(&a);
        assert_eq!(b_shift, qr.0, "shift = {}, b = {:?}", shift, b.to_string_with_radix(16));
        assert_eq!(b_mod, qr.1, "shift = {}, b = {:?}", shift, b.to_string_with_radix(16));

        qr = b.divide_and_remainder(&a.negate());
        assert_eq!(b_shift.negate(), qr.0, "shift = {}, b = {:?}", shift, b.to_string_with_radix(16));
        assert_eq!(b_mod, qr.1, "shift = {}, b = {:?}", shift, b.to_string_with_radix(16));

        qr = b.negate().divide_and_remainder(&a);
        assert_eq!(b_shift.negate(), qr.0, "shift = {}, b = {:?}", shift, b.to_string_with_radix(16));
        assert_eq!(b_mod.negate(), qr.1, "shift = {}, b = {:?}", shift, b.to_string_with_radix(16));

        qr = b.negate().divide_and_remainder(&a.negate());
        assert_eq!(b_shift, qr.0, "shift = {}, b = {:?}", shift, b.to_string_with_radix(16));
        assert_eq!(b_mod.negate(), qr.1, "shift = {}, b = {:?}", shift, b.to_string_with_radix(16));
    }
}

#[test]
fn test_flip_bit() {
    let mut random = DefaultRandomSource::default();
    for _ in 0..10 {
        let a = BigInteger::with_random_certainty(128, 0, &mut random);
        let b = a.clone();

        for _ in 0..100 {
            // Note: Intentionally greater than initial size
            let pos = std::random::random::<usize>() % 256;

            let a = a.flip_bit(pos);
            let b = if b.test_bit(pos) {
                b.clear_bit(pos)
            } else {
                b.set_bit(pos)
            };

            assert_eq!(a, b);
        }
    }

    for i in 0..100 {
        let pow2 = (*ONE).shift_left(i);
        let minus_pow2 = pow2.negate();

        assert_eq!((*ZERO), pow2.flip_bit(i as usize));
        assert_eq!(
            minus_pow2.shift_left(1),
            minus_pow2.flip_bit(i as usize),
            "i = {}",
            i
        );

        let big_i = BigInteger::with_i32(i as i32);
        let neg_i = big_i.negate();

        for j in 0..10 {
            assert_eq!(
                big_i.xor(&(*ONE).shift_left(j)),
                big_i.flip_bit(j as usize),
                "i = {}, j = {}",
                i,
                j
            );
            assert_eq!(
                neg_i.xor(&(*ONE).shift_left(j)),
                neg_i.flip_bit(j as usize),
                "i = {}, j = {}",
                i,
                j
            );
        }
    }
}

#[test]
fn test_gcd() {
    let mut random = DefaultRandomSource::default();
    for _ in 0..10 {
        let fac = BigInteger::with_random(32, &mut random).add(&(*TWO));
        let p1 = BigInteger::with_probable_prime(63, &mut random);
        let p2 = BigInteger::with_probable_prime(64, &mut random);
        let gcd = fac.multiply(&p1).gcd(&fac.multiply(&p2));

        assert_eq!(fac, gcd);
    }
}

#[test]
fn tst_get_lowest_set_bit() {
    let mut random = DefaultRandomSource::default();
    for i in 1..=100 {
        let test = BigInteger::with_random_certainty(i + 1, 0, &mut random);
        let bit1 = test.get_lowest_set_bit();
        assert_eq!(test, test.shift_right(bit1).shift_left(bit1));
        let bit2 = test.shift_left(i as i32 + 1).get_lowest_set_bit();
        assert_eq!(i as i32 + 1, bit2 - bit1);
        let bit3 = test.shift_left(3 * i as i32).get_lowest_set_bit();
        assert_eq!(3 * i as i32, bit3 - bit1);
    }
}

#[test]
fn test_i32_value() {
    let tests = [i32::MIN, -1234, -10, -1, 0, !0, 1, 10, 5678, i32::MAX];
    for i in tests.iter() {
        let a = BigInteger::with_i32(*i);
        assert_eq!(*i, a.get_i32_value(), "i = {}", i);
    }
}

#[test]
fn test_is_probable_prime() {
    assert!(!&(*ZERO).is_probable_prime(100));
    assert!(&(*ZERO).is_probable_prime(0));
    assert!(&(*ZERO).is_probable_prime(-10));
    assert!(!&(*MINUS_ONE).is_probable_prime(100));
    assert!(&(*MINUS_TWO).is_probable_prime(100));
    assert!(BigInteger::with_i32(-17).is_probable_prime(100));
    assert!(BigInteger::with_i32(67).is_probable_prime(100));
    assert!(BigInteger::with_i32(773).is_probable_prime(100));

    for p in FIRST_PRIMES {
        assert!(BigInteger::with_i32(p).is_probable_prime(100));
        assert!(BigInteger::with_i32(-p).is_probable_prime(100));
    }

    for c in NON_PRIMES {
        assert!(!BigInteger::with_i32(c).is_probable_prime(100));
        assert!(!BigInteger::with_i32(-c).is_probable_prime(100));
    }

    for e in MERSENNE_PRIME_EXPONENTS {
        assert!(&(*TWO).pow(e as u32).subtract(&(*ONE)).is_probable_prime(100), "e = {}", e);
        assert!(&(*TWO).pow(e as u32).subtract(&(*ONE)).negate().is_probable_prime(100));
    }

    for e in NON_PRIME_EXPONENTS {
        assert!(
            !(&(*TWO)
                .pow(e as u32)
                .subtract(&(*ONE))
                .is_probable_prime(100))
        );
        assert!(
            !(&(*TWO)
                .pow(e as u32)
                .subtract(&(*ONE))
                .negate()
                .is_probable_prime(100))
        );
    }
}

#[test]
fn test_long_value() {
    let tests = [i64::MIN, -1234, -10, -1, 0, !0, 1, 10, 5678, i64::MAX];
    for i in tests.iter() {
        let a = BigInteger::with_i64(*i);
        assert_eq!(*i, a.get_i64_value(), "i = {}", i);
    }
}

#[test]
fn test_max() {
    for i in -10..=10 {
        for j in -10..=10 {
            let a = BigInteger::with_i32(j);
            let b = BigInteger::with_i32(i);
            assert_eq!(a.max(&b), BigInteger::with_i32(i.max(j)));
        }
    }
}

#[test]
fn test_min() {
    for i in -10..=10 {
        for j in -10..=10 {
            let a = BigInteger::with_i32(j);
            let b = BigInteger::with_i32(i);
            assert_eq!(a.min(&b), BigInteger::with_i32(i.min(j)));
        }
    }
}

#[test]
fn test_mod() {
    // TODO Basic tests
    let mut random = DefaultRandomSource::default();
    for _ in 0..100 {
        let diff = std::random::random::<usize>() % 25;
        let a = BigInteger::with_random_certainty(100 - diff, 0, &mut random);
        let b = BigInteger::with_random_certainty(100 + diff, 0, &mut random);
        let c = BigInteger::with_random_certainty(10 + diff, 0, &mut random);

        let d = a.multiply(&b).add(&c);
        let e = d.remainder(&a);
        assert_eq!(c, e);

        let pow2 = (*ONE).shift_left((std::random::random::<usize>() % 128) as i32);
        assert_eq!(b.and(&pow2.subtract(&(*ONE))), b.r#mod(&pow2));
    }
}

#[test]
fn test_mod_inverse() {
    let mut random = DefaultRandomSource::default();
    for i in 0..10 {
        let p = BigInteger::with_probable_prime(64 + i, &mut random);
        let q = BigInteger::with_random(63 + i, &mut random).add(&(*ONE));
        let inv = q.mod_inverse(&p);
        let inv2 = inv.mod_inverse(&p);

        assert_eq!(q, inv2);
        assert_eq!((*ONE), q.multiply(&inv).r#mod(&p));
    }

    // ModInverse a power of 2 for a range of powers
    for i in 1..=128 {
        let m = (*ONE).shift_left(i);
        let d = BigInteger::with_random(i as usize , &mut random).set_bit(0);
        let x = d.mod_inverse(&m);
        let check = x.multiply(&d).r#mod(&m);

        assert_eq!((*ONE), check);
    }
}

#[test]
#[should_panic(expected = "modulus must be positive")]
fn test_mod_pow_01() {
    (*TWO).mod_pow(&(*ONE), &(*ZERO));
}

#[test]
fn test_mod_pow_02() {
    assert_eq!((*ZERO), (*ZERO).mod_pow(&(*ZERO), &(*ONE)));
    assert_eq!((*ONE), (*ZERO).mod_pow(&(*ZERO), &(*TWO)));
    assert_eq!((*ZERO), (*TWO).mod_pow(&(*ONE), &(*ONE)));
    assert_eq!((*ONE), (*TWO).mod_pow(&(*ZERO), &(*TWO)));

    let mut random = DefaultRandomSource::default();
    for i in 0..100 {
        let m = BigInteger::with_probable_prime(10 + i, &mut random);
        let x = BigInteger::with_random(*m.get_bit_length() - 1, &mut random);
        assert_eq!(x, x.mod_pow(&m, &m));
    }
}

#[test]
fn test_multiply() {
    let one = &(*ONE);
    assert_eq!(one, &one.negate().multiply(&one.negate()));

    let mut random = DefaultRandomSource::default();
    for _ in 0..100 {
        let a_len = 64 + std::random::random::<usize>() % 64;
        let b_len = 64 + std::random::random::<usize>() % 64;

        let a = BigInteger::with_random(a_len, &mut random).set_bit(a_len);
        let b = BigInteger::with_random(b_len, &mut random).set_bit(b_len);
        let c = BigInteger::with_random(32, &mut random);

        let ab = a.multiply(&b);
        let bc = b.multiply(&c);

        // println!("a = {:?}", a);
        // println!("b = {:?}", b);
        // println!("c = {:?}", c);
        // println!("ab = {:?}", ab);
        // println!("bc = {:?}", bc);

        assert_eq!(ab.add(&bc), a.add(&c).multiply(&b));
        assert_eq!(ab.subtract(&bc), a.subtract(&c).multiply(&b));
    }

    // Special tests for power of two since uses different code path internally
    for _ in 0..100 {
        let shift = (std::random::random::<usize>() % 64) as i32;
        let a = one.shift_left(shift);
        let b = BigInteger::with_random(64 + std::random::random::<usize>() % 64, &mut random);
        let b_shift = b.shift_left(shift);

        assert_eq!(b_shift, a.multiply(&b));
        assert_eq!(b_shift.negate(), a.multiply(&b.negate()));
        assert_eq!(b_shift.negate(), a.negate().multiply(&b));
        assert_eq!(b_shift, a.negate().multiply(&b.negate()));

        assert_eq!(b_shift, b.multiply(&a));
        assert_eq!(b_shift.negate(), b.multiply(&a.negate()));
        assert_eq!(b_shift.negate(), b.negate().multiply(&a));
        assert_eq!(b_shift, b.negate().multiply(&a.negate()));
    }
}

#[test]
fn test_negate() {
    for i in -10..=10 {
        let a = BigInteger::with_i32(-i);
        let b = BigInteger::with_i32(i);
        let c = b.negate();
        assert_eq!(a, c, "Problem {} NEGATE should be {}", i, -i);
    }
}

#[test]
fn test_next_probable_prime() {
    let mut random = DefaultRandomSource::default();
    let first_prime = BigInteger::with_probable_prime(32, &mut random);
    let next_prime = first_prime.next_probable_prime();

    assert!(first_prime.is_probable_prime(10));
    assert!(next_prime.is_probable_prime(10));

    let check = first_prime.add(&(*ONE));
    while check < next_prime {
        assert!(!check.is_probable_prime(10));
        check.add(&(*ONE));
    }
}

#[test]
fn test_not() {
    for i in -10..=10 {
        let a = BigInteger::with_i32(!i);
        let b = BigInteger::with_i32(i);
        let c = b.not();
        assert_eq!(a, c, "Problem {} NOT should be {}", i, !i);
    }
}

#[test]
fn test_or() {
    for i in -10..=10 {
        for j in 1..=10 {
            let a = BigInteger::with_i32(i | j);
            let b = BigInteger::with_i32(i);
            let c = BigInteger::with_i32(j);
            let d = b.or(&c);
            assert_eq!(a, d, "Problem {} OR {} should be {}", i, j, i | j);
        }
    }
}

#[test]
fn test_pow() {
    assert_eq!(*ONE, (*ZERO).pow(0));
    assert_eq!(*ZERO, (*ZERO).pow(123));
    assert_eq!(*ONE, (*ONE).pow(0));
    assert_eq!(*ONE, (*ONE).pow(123));

    assert_eq!((*TWO).pow(147), (*ONE).shift_left(147));
    assert_eq!((*ONE).shift_left(7).pow(11), (*ONE).shift_left(77));

    let n = BigInteger::with_string("1234567890987654321").expect("error");
    let mut result = (*ONE).clone();
    for i in 0..10 {
        assert_eq!(result, n.pow(i), "i = {}", i);
        result = result.multiply(&n);
    }
}

#[test]
fn test_remainder() {
    // TODO Basic tests
    for rep in 0..10 {
        let a =
            BigInteger::with_random_certainty(100 - rep, 0, &mut DefaultRandomSource::default());
        let b =
            BigInteger::with_random_certainty(100 + rep, 0, &mut DefaultRandomSource::default());
        let c = BigInteger::with_random_certainty(10 + rep, 0, &mut DefaultRandomSource::default());
        let d = a.multiply(&b).add(&c);
        let e = d.remainder(&a);
        let f = d.divide(&a);

        // println!("a = {:?}", a);
        // println!("b = {:?}", b);
        // println!("c = {:?}", c);
        // println!("d = {:?}", d);
        // println!("e = {:?}", e);
        // println!("f = {:?}", f);
        assert_eq!(b, f);
        assert_eq!(c, e);
    }
}

#[test]
fn test_set_bit() {
    assert_eq!(&(*ONE), &(*ZERO).set_bit(0));
    assert_eq!(&(*ONE), &(*ONE).set_bit(0));
    assert_eq!(&(*THREE), &(*TWO).set_bit(0));

    assert_eq!(&(*TWO), &(*ZERO).set_bit(1));
    assert_eq!(&(*THREE), &(*ONE).set_bit(1));
    assert_eq!(&(*TWO), &(*TWO).set_bit(1));

    // TODO Tests for setting bits in negative numbers

    // TODO Tests for setting extended bits
    let mut random = DefaultRandomSource::default();
    for _ in 0..10 {
        let n = BigInteger::with_random(128, &mut random);

        for _ in 0..10 {
            let pos = std::random::random::<usize>() % 128;
            let m = n.set_bit(pos);
            let test = m.shift_right(pos as i32).remainder(&(*TWO)) == *ONE;
            assert!(test);
        }
    }
}

#[test]
fn test_shift_left() {
    let mut random = DefaultRandomSource::default();
    for i in 0..100 {
        let shift = std::random::random::<usize>() % 128;

        let a = BigInteger::with_random(128 + i, &mut random);
        a.get_bit_count();

        let neg_a = a.negate();
        neg_a.get_bit_count();

        let b = a.shift_left(shift as i32);
        let c = neg_a.shift_left(shift as i32);

        assert_eq!(
            a.get_bit_count(),
            b.get_bit_count(),
            "1. i = {}, shift = {}",
            i,
            shift
        );
        assert_eq!(
            *neg_a.get_bit_count() + shift,
            *c.get_bit_count(),
            "2. i = {}, shift = {}",
            i,
            shift
        );
        assert_eq!(*a.get_bit_length() + shift, *b.get_bit_length());
        assert_eq!(*neg_a.get_bit_length() + shift, *c.get_bit_length());

        let mut j = 0usize;
        while j < shift {
            assert!(!b.test_bit(j));
            j += 1;
        }
        while j < (*b.get_bit_length()) {
            assert_eq!(a.test_bit(j - shift), b.test_bit(j));
            j += 1;
        }
    }
}

#[test]
fn test_shift_right() {
    let mut random = DefaultRandomSource::default();
    for i in 0..10 {
        let shift = std::random::random::<usize>() % 128;
        let a = BigInteger::with_random(256 + i, &mut random);
        let b = a.shift_right(shift as i32);

        assert_eq!(*a.get_bit_length() - shift, *b.get_bit_length());

        for j in 0..(*b.get_bit_length()) {
            assert_eq!(a.test_bit(j + shift), b.test_bit(j));
        }
    }
}

#[test]
fn test_sign_value() {
    for i in -10..=10 {
        let a = BigInteger::with_i32(i);
        let b = if i < 0 {
            -1
        } else if i > 0 {
            1
        } else {
            0
        };
        assert_eq!(b, a.get_sign_value());
    }
}

#[test]
fn test_subtract() {
    for i in -10..=10 {
        for j in -10..=10 {
            let a = BigInteger::with_i32(i - j);
            let b = BigInteger::with_i32(i);
            let c = BigInteger::with_i32(j);
            let d = b.subtract(&c);
            assert_eq!(a, d, "Problem {} SUBTRACT {} should be {}", i, j, i - j);
        }
    }

    let mut random = DefaultRandomSource::default();
    for _ in 0..10 {
        let a = BigInteger::with_random_certainty(128, 0, &mut random);
        let b = BigInteger::with_random_certainty(128, 0, &mut random);
        let c = a.subtract(&b);
        let d = b.subtract(&a);
        assert_eq!(c.abs(), d.abs());
    }
}

#[test]
fn test_test_bit() {
    let mut random = DefaultRandomSource::default();
    for _ in 0..10 {
        let n = BigInteger::with_random(128, &mut random);
        assert!(!n.test_bit(128));
        assert!(n.negate().test_bit(128));

        for _ in 0..10 {
            let pos = std::random::random::<usize>() % 128;
            let test = n.shift_right(pos as i32).remainder(&(*TWO)) == *ONE;
            assert_eq!(test, n.test_bit(pos));
        }
    }
}

#[test]
fn test_to_vec() {
    let z = &(*ZERO).to_vec();
    assert_eq!(z.len(), 1);
    assert_eq!(z[0], 0);

    let mut random = DefaultRandomSource::default();
    for i in 16..=48 {
        let x = BigInteger::with_random(i, &mut random).set_bit(i - 1);
        let b = x.to_vec();
        assert_eq!(b.len(), i / 8 + 1);
        let y = BigInteger::with_buffer(&b).expect("Failed to create BigInteger from buffer");
        assert_eq!(x, y);

        let x = x.negate();
        let b = x.to_vec();
        assert_eq!(b.len(), i / 8 + 1);
        let y = BigInteger::with_buffer(&b).expect("Failed to create BigInteger from buffer");
        assert_eq!(x, y);
    }
}

#[test]
fn test_to_vec_unsigned() {
    let z = &(*ZERO).to_vec_unsigned();
    assert_eq!(z.len(), 0);

    let mut random = DefaultRandomSource::default();
    for i in 16..=48 {
        let mut x = BigInteger::with_random(i, &mut random).set_bit(i - 1);
        let mut b = x.to_vec_unsigned();
        assert_eq!(b.len(), (i + 7) / 8);
        let mut y =
            BigInteger::with_sign_buffer(1, &b).expect("Failed to create BigInteger from buffer");
        assert_eq!(x, y);

        x = x.negate();
        b = x.to_vec_unsigned();
        assert_eq!(b.len(), i / 8 + 1);
        y = BigInteger::with_buffer(&b).expect("Failed to create BigInteger from buffer");
        assert_eq!(x, y);
    }
}

#[test]
fn test_to_string() {
    let s = "1234567890987654321";
    assert_eq!(s, &BigInteger::with_string(s).unwrap().to_string());
    assert_eq!(
        s,
        &BigInteger::with_string_radix(s, 10)
            .unwrap()
            .to_string_with_radix(10)
    );
    assert_eq!(
        s,
        &BigInteger::with_string_radix(s, 16)
            .unwrap()
            .to_string_with_radix(16)
    );

    let mut random = DefaultRandomSource::default();
    for i in 0..100 {
        let left = BigInteger::with_random(i, &mut random);
        {
            let right = BigInteger::with_string_radix(&left.to_string_with_radix(2), 2).unwrap();
            assert_eq!(
            left,
            right,
            "radix = 2, i = {}, \n left: n2 = {}, n8 = {}, n10 = {}, n16 = {} \n right: n2 = {}, n8 = {}, n10 = {}, n16 = {}",
            i,
            left.to_string_with_radix(2),
            left.to_string_with_radix(8),
            left.to_string_with_radix(10),
            left.to_string_with_radix(16),
            right.to_string_with_radix(2),
            right.to_string_with_radix(8),
            right.to_string_with_radix(10),
            right.to_string_with_radix(16)
        );
        }
        {
            let right = BigInteger::with_string_radix(&left.to_string_with_radix(8), 8).unwrap();
            assert_eq!(
            left,
            right,
            "radix = 8, i = {}, \n left: n2 = {}, n8 = {}, n10 = {}, n16 = {} \n right: n2 = {}, n8 = {}, n10 = {}, n16 = {}",
            i,
            left.to_string_with_radix(2),
            left.to_string_with_radix(8),
            left.to_string_with_radix(10),
            left.to_string_with_radix(16),
            right.to_string_with_radix(2),
            right.to_string_with_radix(8),
            right.to_string_with_radix(10),
            right.to_string_with_radix(16)
        );
        }
        {
            let right = BigInteger::with_string_radix(&left.to_string_with_radix(10), 10).unwrap();
            assert_eq!(
            left,
            right,
            "radix = 10, i = {}, \n left: n2 = {}, n8 = {}, n10 = {}, n16 = {} \n right: n2 = {}, n8 = {}, n10 = {}, n16 = {}",
            i,
            left.to_string_with_radix(2),
            left.to_string_with_radix(8),
            left.to_string_with_radix(10),
            left.to_string_with_radix(16),
            right.to_string_with_radix(2),
            right.to_string_with_radix(8),
            right.to_string_with_radix(10),
            right.to_string_with_radix(16)
        );
        }
        let right = BigInteger::with_string_radix(&left.to_string_with_radix(16), 16).unwrap();
        assert_eq!(
            left,
            right,
            "radix = 2, i = {}, \n left: n2 = {}, n8 = {}, n10 = {}, n16 = {} \n right: n2 = {}, n8 = {}, n10 = {}, n16 = {}",
            i,
            left.to_string_with_radix(2),
            left.to_string_with_radix(8),
            left.to_string_with_radix(10),
            left.to_string_with_radix(16),
            right.to_string_with_radix(2),
            right.to_string_with_radix(8),
            right.to_string_with_radix(10),
            right.to_string_with_radix(16)
        );
    }

    // Radix version
    let radices = [2, 8, 10, 16];
    let trials = 256;

    let mut tests = vec![(*ZERO).clone(); trials];
    for i in 0..trials {
        let len = std::random::random::<usize>() % (i + 1);
        tests[i] = BigInteger::with_random(len, &mut random);
    }

    for radix in radices {
        for i in 0..trials {
            let n1 = &tests[i];
            let str = n1.to_string_with_radix(radix);
            let n2 = BigInteger::with_string_radix(&str, radix).unwrap();
            assert_eq!(n1, &n2);
        }
    }
}

#[test]
fn test_value_of() {
    assert_eq!(-1, BigInteger::with_i32(-1).get_sign_value());
    assert_eq!(0, BigInteger::with_i32(0).get_sign_value());
    assert_eq!(1, BigInteger::with_i32(1).get_sign_value());

    for i in -5..5 {
        let a = BigInteger::with_i32(i);
        assert_eq!(i, a.get_i32_value());
    }
}

#[test]
fn test_xor() {
    for i in -10..=10 {
        for j in -10..=10 {
            let a = BigInteger::with_i32(i ^ j);
            let b = BigInteger::with_i32(i);
            let c = BigInteger::with_i32(j);
            let d = b.xor(&c);
            assert_eq!(a, d, "Problem {} XOR {} should be {}", i, j, i ^ j);
        }
    }
}
