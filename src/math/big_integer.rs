use std::fmt::Debug;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::random::{DefaultRandomSource, RandomSource};
use std::sync::{Arc, LazyLock, OnceLock};

use crate::math::raw::internal_mod::{inverse_u32, inverse_u64};
use crate::util::pack::{be_to_u32_buffer, be_to_u32_low, le_to_u32_low, u32_to_be_bytes};
use crate::{BcError, Result};

const IMASK: i64 = 0xFFFFFFFF;
const UIMASK: u64 = 0xFFFFFFFF;

const C_ZERO_MAGNITUDE: Vec<u32> = Vec::new();
const C_BYTES_PRE_INT: usize = size_of::<u32>();
const BITS_PER_INT: usize = C_BYTES_PRE_INT * 8;
const BITS_PER_BYTE: usize = size_of::<u8>() * 8;

/// Represents the constant integer value 0 as a BigInteger instance.
pub static ZERO: LazyLock<BigInteger> =
    LazyLock::new(|| BigInteger::new(0, Arc::new(C_ZERO_MAGNITUDE)));

macro_rules! create_static_big_integer {
    ($value:expr) => {
        LazyLock::new(|| BigInteger::new(1, Arc::new(vec![$value])))
    };
    ($sign:expr, $value:expr) => {
        LazyLock::new(|| BigInteger::new($sign, Arc::new(vec![$value])))
    };
}

/// Represents the constant integer value 1 as a BigInteger instance.
///
/// This is a statically allocated constant using LazyLock for lazy initialization.
/// The actual BigInteger object is only created upon first access.
///
/// # Examples
/// ```
/// use bc_rust::math::big_integer::ONE;
///
/// assert_eq!(&(*ONE).to_string(), "1");
/// ```
pub static ONE: LazyLock<BigInteger> = create_static_big_integer!(1);
/// Represents the constant integer value 2 as a BigInteger instance.
pub static TWO: LazyLock<BigInteger> = create_static_big_integer!(2);
/// Represents the constant integer value 3 as a BigInteger instance.
pub static THREE: LazyLock<BigInteger> = create_static_big_integer!(3);
/// Represents the constant integer value 4 as a BigInteger instance.
pub static FOUR: LazyLock<BigInteger> = create_static_big_integer!(4);
/// Represents the constant integer value 5 as a BigInteger instance.
pub static FIVE: LazyLock<BigInteger> = create_static_big_integer!(5);
/// Represents the constant integer value 6 as a BigInteger instance.
pub static SIX: LazyLock<BigInteger> = create_static_big_integer!(6);
/// Represents the constant integer value 7 as a BigInteger instance.
pub static SEVEN: LazyLock<BigInteger> = create_static_big_integer!(7);
/// Represents the constant integer value 8 as a BigInteger instance.
pub static EIGHT: LazyLock<BigInteger> = create_static_big_integer!(8);
/// Represents the constant integer value 9 as a BigInteger instance.
pub static NINE: LazyLock<BigInteger> = create_static_big_integer!(9);
/// Represents the constant integer value 10 as a BigInteger instance.
pub static TEN: LazyLock<BigInteger> = create_static_big_integer!(10);
static SMALL_CONSTANTS: LazyLock<[BigInteger; 17]> = LazyLock::new(|| {
    let result: [BigInteger; 17] = [
        (*ZERO).clone(),
        (*ONE).clone(),
        (*TWO).clone(),
        (*THREE).clone(),
        (*FOUR).clone(),
        (*FIVE).clone(),
        (*SIX).clone(),
        (*SEVEN).clone(),
        (*EIGHT).clone(),
        (*NINE).clone(),
        (*TEN).clone(),
        BigInteger::new(1, Arc::new(vec![11])),
        BigInteger::new(1, Arc::new(vec![12])),
        BigInteger::new(1, Arc::new(vec![13])),
        BigInteger::new(1, Arc::new(vec![14])),
        BigInteger::new(1, Arc::new(vec![15])),
        BigInteger::new(1, Arc::new(vec![16])),
    ];
    result
});

// TODO Parse radix-2 64 bits at a time and radix-8 63 bits at a time
const CHUNK_2: u32 = 1;
const CHUNK_8: u32 = 1;
const CHUNK_10: u32 = 19;
const CHUNK_16: u32 = 16;

static RADIX_2: LazyLock<BigInteger> = LazyLock::new(|| (*SMALL_CONSTANTS)[2].clone());
static RADIX_2E: LazyLock<BigInteger> = LazyLock::new(|| (*RADIX_2).pow(CHUNK_2).unwrap());
static RADIX_8: LazyLock<BigInteger> = LazyLock::new(|| (*SMALL_CONSTANTS)[8].clone());
static RADIX_8E: LazyLock<BigInteger> = LazyLock::new(|| (*RADIX_8).pow(CHUNK_8).unwrap());
static RADIX_10: LazyLock<BigInteger> = LazyLock::new(|| (*SMALL_CONSTANTS)[10].clone());
static RADIX_10E: LazyLock<BigInteger> = LazyLock::new(|| (*RADIX_10).pow(CHUNK_10).unwrap());
static RADIX_16: LazyLock<BigInteger> = LazyLock::new(|| (*SMALL_CONSTANTS)[16].clone());
static RADIX_16E: LazyLock<BigInteger> = LazyLock::new(|| (*RADIX_16).pow(CHUNK_16).unwrap());

/// `BigInteger` is a structure for representing large integers.
#[derive(Clone)]
pub struct BigInteger {
    sign: i32,
    magnitude: Arc<Vec<u32>>,
    bits: OnceLock<usize>,
    bit_length: OnceLock<usize>,
}

impl BigInteger {
    fn new(sign: i32, magnitude: Arc<Vec<u32>>) -> Self {
        Self {
            sign,
            magnitude,
            bits: OnceLock::new(),
            bit_length: OnceLock::new(),
        }
    }

    /// Creates a `BigInteger` from a string.
    ///
    /// # Parameters
    ///
    /// * `str` - A string representing the integer.
    ///
    /// # Errors
    ///
    /// Returns `Error` if the string is empty.
    pub fn with_string(str: &str) -> Result<Self> {
        Self::with_string_radix(str, 10)
    }

    /// Creates a `BigInteger` from a string with the specified radix.
    ///
    /// # Parameters
    ///
    /// * `str` - A string representing the integer.
    /// * `radix` - The radix/base. Must be 2, 8, 10, or 16.
    ///
    /// # Errors
    ///
    /// Returns `Error` if the string is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use bc_rust::math::BigInteger;
    /// use std::io::Error;
    ///
    /// let result_radix2 = BigInteger::with_string_radix("1010", 2).expect("Error");
    /// let result_radix8 = BigInteger::with_string_radix("12", 8).expect("Error");
    /// let result_radix10 = BigInteger::with_string_radix("10", 10).expect("Error");
    /// let result_radix16 = BigInteger::with_string_radix("A", 16).expect("Error");
    ///
    /// assert_eq!(result_radix2, result_radix8);
    /// assert_eq!(result_radix8, result_radix10);
    /// assert_eq!(result_radix10, result_radix16);
    /// ```
    pub fn with_string_radix(str: &str, radix: u32) -> Result<Self> {
        anyhow::ensure!(
            !str.is_empty(),
            BcError::invalid_argument("Zero length BigInteger", "str")
        );

        let chunk: u32;
        let r: &BigInteger;
        let re: &BigInteger;
        match radix {
            2 => {
                chunk = CHUNK_2;
                r = &(*RADIX_2);
                re = &(*RADIX_2E);
            }
            8 => {
                chunk = CHUNK_8;
                r = &(*RADIX_8);
                re = &(*RADIX_8E);
            }
            10 => {
                chunk = CHUNK_10;
                r = &(*RADIX_10);
                re = &(*RADIX_10E);
            }
            16 => {
                chunk = CHUNK_16;
                r = &(*RADIX_16);
                re = &(*RADIX_16E);
            }
            _ => {
                anyhow::bail!(BcError::invalid_argument("Invalid radix", "radix"));
            }
        };

        let mut index = 0usize;
        let mut sign = 1;
        if let Some(ch) = str.chars().next() {
            if ch == '-' {
                sign = -1;
                index += 1;
            }
        }

        // strip leading zeros from the string str
        let mut chars = str.chars().skip(index);
        while let Some(ch) = chars.next() {
            if let Some(0) = ch.to_digit(radix) {
                index += 1;
            } else {
                break;
            }
        }
        if index >= str.chars().count() {
            return Ok((*ZERO).clone());
        }

        //////
        // could we work out the max number of ints required to store
        // str.Length digits in the given base, then allocate that
        // storage in one hit?, then Generate the magnitude in one hit too?
        //////
        let mut b = (*&ZERO).clone();
        let mut next = index + chunk as usize;
        if next <= str.chars().count() {
            loop {
                let s = str.get(index..next).unwrap();
                let i = u64::from_str_radix(&s, radix)?;
                let bi = BigInteger::with_u64(i);
                match radix {
                    2 => {
                        anyhow::ensure!(
                            i < 2,
                            BcError::invalid_argument(
                                &format!("Bad character in radix 2 string: {}", s),
                                "radix"
                            )
                        );
                        b = b.shift_left(1);
                    }
                    8 => {
                        anyhow::ensure!(
                            i < 8,
                            BcError::invalid_argument(
                                &format!("Bad character in radix 8 string: {}", s),
                                "radix"
                            )
                        );
                        b = b.shift_left(3);
                    }
                    16 => {
                        b = b.shift_left(64);
                    }
                    _ => {
                        b = b.multiply(&re);
                    }
                }
                b = b.add(&bi);
                index = next;
                next += chunk as usize;

                if next <= str.chars().count() {
                    // nothing
                } else {
                    break;
                }
            }
        }

        if index < str.chars().count() {
            let s = str.get(index..).unwrap();
            let i = u64::from_str_radix(&s, radix)?;
            let bi = BigInteger::with_u64(i);
            if b.sign > 0 {
                if radix == 2 {
                    // NB: Can't reach here since we are parsing one char at a time
                    debug_assert!(false);
                    // TODO Parse all bits at once
                    // b = b.ShiftLeft(s.Length);
                } else if radix == 8 {
                    // NB: Can't reach here since we are parsing one char at a time
                    debug_assert!(false);
                    // TODO Parse all bits at once
                    // b = b.ShiftLeft(s.Length * 3);
                } else if radix == 16 {
                    b = b.shift_left((s.chars().count() as i32) << 2);
                } else {
                    b = b.multiply(&r.pow(s.chars().count() as u32)?);
                }

                b = b.add(&bi);
            } else {
                b = bi;
            }
        }
        if sign < 0 {
            b = b.negate();
        }
        Ok(b)
    }

    pub fn with_sign_buffer(sign: i32, buffer: &[u8]) -> Result<Self> {
        Self::with_sign_buffer_big_endian(sign, buffer, true)
    }

    /// # Arguments
    /// * `sign` - The sign of the integer.
    /// * `buffer` - The buffer containing the integer.
    /// * `big_endian` - The endianness of the buffer.
    /// # Returns
    /// A `BigInteger` instance.
    /// # Errors
    /// Returns `Error` if the sign is invalid.
    pub fn with_sign_buffer_big_endian(sign: i32, buffer: &[u8], big_endian: bool) -> Result<Self> {
        anyhow::ensure!(
            sign == 0 || sign == 1 || sign == -1,
            BcError::invalid_argument("Invalid sign value", "sign")
        );
        if sign == 0 {
            return Ok((*ZERO).clone());
        }
        let magnitude = if big_endian {
            make_magnitude_be(buffer)
        } else {
            make_magnitude_le(buffer)
        };
        Ok(Self::new(sign, Arc::new(magnitude)))
    }

    pub fn with_buffer(buffer: &[u8]) -> Self {
        Self::with_buffer_big_endian(buffer, true)
    }

    pub fn with_buffer_big_endian(buffer: &[u8], big_endian: bool) -> Self {
        let result = if big_endian {
            init_be(buffer)
        } else {
            init_le(buffer)
        };
        Self::new(result.1, Arc::new(result.0))
    }

    pub fn with_i32(value: i32) -> BigInteger {
        if value >= 0 {
            if (value as usize) < SMALL_CONSTANTS.len() {
                return SMALL_CONSTANTS[value as usize].clone();
            }
            return create_value_of_u32(value as u32);
        }
        if value == i32::MIN {
            return create_value_of_u32(!value as u32).not();
        }
        BigInteger::with_i32(-value).negate()
    }
    pub fn with_u32(value: u32) -> BigInteger {
        return BigInteger::new(1, Arc::new(vec![value]));
    }
    pub fn with_u64(value: u64) -> BigInteger {
        let msw = (value >> 32) as u32;
        let lsw = value as u32;
        if msw == 0 {
            return BigInteger::with_u32(lsw);
        }
        return BigInteger::new(1, Arc::new(vec![msw, lsw]));
    }
    pub fn with_i64(value: i64) -> BigInteger {
        if value >= 0 {
            if value < SMALL_CONSTANTS.len() as i64 {
                return SMALL_CONSTANTS[value as usize].clone();
            }
            return BigInteger::with_u64(value as u64);
        }

        if value == i64::MIN {
            return BigInteger::with_u64(!value as u64).not();
        }

        return BigInteger::with_i64(-value).negate();
    }
    pub fn with_random(size_in_bits: usize, random: &mut dyn RandomSource) -> BigInteger {
        if size_in_bits == 0 {
            return (*ZERO).clone();
        }
        let n_bytes = get_bytes_length(size_in_bits);
        let mut b = vec![0u8; n_bytes];
        random.fill_bytes(&mut b[..]);
        let x_bits = (BITS_PER_BYTE * n_bytes) - size_in_bits;
        b[0] &= (255 >> x_bits) as u8;
        let magnitude = make_magnitude_be(&b);
        let sign = if magnitude.len() < 1 { 0 } else { 1 };
        BigInteger::new(sign, Arc::new(magnitude))
    }

    pub fn with_random_certainty(
        size_in_bits: usize,
        certainty: i32,
        random: &mut dyn RandomSource,
    ) -> Result<Self> {
        anyhow::ensure!(
            size_in_bits >= 2,
            BcError::invalid_argument("size_in_bits < 2", "size_in_bits")
        );
        if size_in_bits == 2 {
            return if std::random::random::<u32>() % 2 == 0 {
                Ok((*TWO).clone())
            } else {
                Ok((*THREE).clone())
            };
        }
        let n_bytes = get_bytes_length(size_in_bits);
        let mut b = vec![0u8; n_bytes];
        let x_bits = BITS_PER_BYTE * n_bytes - size_in_bits;
        let mask = (255 >> x_bits) as u8;
        let lead = (1 << (7 - x_bits)) as u8;
        loop {
            random.fill_bytes(&mut b);

            // strip off any excess bits in the MSB
            b[0] &= mask;

            // ensure the leading bit is 1 (to meet the strength requirement)
            b[0] |= lead;

            // ensure the trailing bit is 1 (i.e. must be odd)
            b[n_bytes - 1] |= 1;
            let mut magnitude = make_magnitude_be(&mut b);
            if certainty < 1 {
                return Ok(BigInteger::new(1, Arc::new(magnitude)));
            }

            let result = BigInteger::new(1, Arc::new(magnitude.clone()));
            if result.check_probable_prime(certainty, random, true)? {
                return Ok(result);
            }
            for j in 1..(magnitude.len() - 1) {
                magnitude[j] ^= std::random::random::<u32>();

                let result = BigInteger::new(1, Arc::new(magnitude.clone()));
                if result.check_probable_prime(certainty, random, true)? {
                    return Ok(result);
                }
            }
        }
    }

    pub fn with_probable_prime(
        bit_length: usize,
        random: &mut dyn RandomSource,
    ) -> Result<BigInteger> {
        Self::with_random_certainty(bit_length, 100, random)
    }

    fn with_check_mag(sign: i32, magnitude: Arc<Vec<u32>>, check_mag: bool) -> Self {
        if check_mag {
            let sub_slice = strip_prefix_value(magnitude.as_slice(), 0x0);
            if sub_slice.is_empty() {
                return Self::new(0, Arc::new(C_ZERO_MAGNITUDE));
            } else {
                return Self::new(sign, Arc::new(sub_slice.to_vec()));
            }
        } else {
            return Self::new(sign, magnitude);
        }
    }
    pub fn abs(&self) -> Self {
        if self.sign >= 0 {
            return Self::new(self.sign, self.magnitude.clone());
        } else {
            self.negate()
        }
    }
    pub fn add(&self, other: &BigInteger) -> BigInteger {
        if self.sign == 0 {
            return other.clone();
        }
        if self.sign == other.sign {
            return self.add_to_magnitude(&other.magnitude);
        }
        if other.sign == 0 {
            return self.clone();
        } else if other.sign < 0 {
            return self.subtract(&other.negate());
        } else {
            return other.subtract(&self.negate());
        }
    }
    fn add_to_magnitude(&self, other: &[u32]) -> BigInteger {
        let (big, small) = if self.magnitude.len() < other.len() {
            (other, self.magnitude.as_slice())
        } else {
            (self.magnitude.as_slice(), other)
        };

        // Conservatively avoid over-allocation when no overflow possible
        let mut limit = u32::MAX;
        if big.len() == small.len() {
            limit -= small[0];
        }

        let possible_over_flow = big[0] >= limit;

        let mut big_copy;
        if possible_over_flow {
            big_copy = vec![0u32; big.len() + 1];
            big_copy[1..(big.len() + 1)].copy_from_slice(big);
        } else {
            big_copy = big.to_vec();
        }
        add_magnitudes(&mut big_copy, small);
        BigInteger::with_check_mag(self.sign, Arc::new(big_copy), possible_over_flow)
    }
    pub fn negate(&self) -> Self {
        if self.sign == 0 {
            Self::new(self.sign, self.magnitude.clone())
        } else {
            Self::new(-self.sign, self.magnitude.clone())
        }
    }
    pub fn subtract(&self, other: &BigInteger) -> BigInteger {
        if other.sign == 0 {
            return self.clone();
        }
        if self.sign == 0 {
            return other.negate();
        }
        if self.sign != other.sign {
            return self.add(&other.negate());
        }
        let compare =
            compare_no_leading_zeros(self.magnitude.as_slice(), other.magnitude.as_slice());
        if compare == 0 {
            return (*ZERO).clone();
        }
        let (bigun, lilun) = if compare < 0 {
            (other, self)
        } else {
            (self, other)
        };
        let magnitude = do_sub_big_lil(bigun.magnitude.as_slice(), lilun.magnitude.as_slice());
        BigInteger::with_check_mag(self.sign * compare, Arc::new(magnitude), true)
    }
    pub fn and(&self, other: &BigInteger) -> BigInteger {
        if self.sign == 0 || other.sign == 0 {
            return (*ZERO).clone();
        }
        let a_mag = if self.sign > 0 {
            self.magnitude.clone()
        } else {
            self.add(&(*ONE)).magnitude.clone()
        };
        let b_mag = if other.sign > 0 {
            other.magnitude.clone()
        } else {
            other.add(&(*ONE)).magnitude.clone()
        };
        let result_neg = self.sign < 0 && other.sign < 0;
        let result_length = std::cmp::max(a_mag.len(), b_mag.len());
        let mut result_mag = vec![0u32; result_length];
        let a_start = (result_mag.len() - a_mag.len()) as isize;
        let b_start = (result_mag.len() - b_mag.len()) as isize;
        for i in 0..result_mag.len() {
            let mut a_word = if (i as isize) >= a_start {
                a_mag[i - a_start as usize]
            } else {
                0u32
            };
            let mut b_word = if (i as isize) >= b_start {
                b_mag[i - b_start as usize]
            } else {
                0u32
            };
            if self.sign < 0 {
                a_word = !a_word;
            }
            if other.sign < 0 {
                b_word = !b_word;
            }
            result_mag[i] = a_word & b_word;
            if result_neg {
                result_mag[i] = !result_mag[i];
            }
        }
        let mut result = BigInteger::with_check_mag(1, Arc::new(result_mag), true);
        // TODO Optimise this case
        if result_neg {
            result = result.not();
        }
        result
    }

    pub fn or(&self, other: &BigInteger) -> BigInteger {
        if self.sign == 0 {
            return other.clone();
        }
        if other.sign == 0 {
            return self.clone();
        }

        let a_mag = if self.sign > 0 {
            self.magnitude.clone()
        } else {
            self.add(&(*ONE)).magnitude.clone()
        };
        let b_mag = if other.sign > 0 {
            other.magnitude.clone()
        } else {
            other.add(&(*ONE)).magnitude.clone()
        };
        let result_neg = self.sign < 0 || other.sign < 0;
        let result_length = std::cmp::max(a_mag.len(), b_mag.len());
        let mut result_mag = vec![0u32; result_length];
        let a_start = (result_mag.len() - a_mag.len()) as isize;
        let b_start = (result_mag.len() - b_mag.len()) as isize;
        for i in 0..result_mag.len() {
            let mut a_word = if (i as isize) >= a_start {
                a_mag[i - a_start as usize]
            } else {
                0u32
            };
            let mut b_word = if (i as isize) >= b_start {
                b_mag[i - b_start as usize]
            } else {
                0u32
            };
            if self.sign < 0 {
                a_word = !a_word;
            }
            if other.sign < 0 {
                b_word = !b_word;
            }
            result_mag[i] = a_word | b_word;
            if result_neg {
                result_mag[i] = !result_mag[i];
            }
        }
        let mut result = BigInteger::with_check_mag(1, Arc::new(result_mag), true);
        // TODO Optimise this case
        if result_neg {
            result = result.not();
        }
        result
    }

    pub fn and_not(&self, other: &BigInteger) -> BigInteger {
        let r1 = other.not();
        let r2 = self.and(&r1);
        r2
    }
    pub fn get_bit_count(&self) -> &usize {
        self.bits.get_or_init(|| {
            if self.sign < 0 {
                return *(self.not().get_bit_count());
            } else {
                let mut sum = 0usize;
                for i in 0..self.magnitude.len() {
                    sum += self.magnitude[i].count_ones() as usize;
                }
                return sum;
            }
        })
    }
    pub fn not(&self) -> BigInteger {
        let r1 = self.inc();
        let r2 = r1.negate();
        r2
    }
    pub fn inc(&self) -> BigInteger {
        if self.sign == 0 {
            return (*ONE).clone();
        }
        if self.sign < 0 {
            return BigInteger::with_check_mag(
                -1,
                Arc::new(do_sub_big_lil(
                    self.magnitude.as_slice(),
                    &(*ONE).magnitude.as_slice(),
                )),
                true,
            );
        }
        return self.add_to_magnitude(&(*ONE).magnitude.as_slice());
    }
    pub fn bit_length(&self) -> &usize {
        self.bit_length.get_or_init(|| {
            if self.sign == 0 {
                0usize
            } else {
                calc_bit_length(self.sign, &self.magnitude)
            }
        })
    }
    pub fn divide(&self, other: &BigInteger) -> Result<BigInteger> {
        anyhow::ensure!(
            other.sign != 0,
            BcError::invalid_argument("divide by zero", "other")
        );
        if self.sign == 0 {
            return Ok((*ZERO).clone());
        }
        if other.quick_pow2_check() {
            let result = self
                .abs()
                .shift_right((other.abs().bit_length() - 1) as i32);
            return if other.sign == self.sign {
                Ok(result)
            } else {
                Ok(result.negate())
            };
        }
        let mut mag = self.magnitude.to_vec();
        Ok(BigInteger::with_check_mag(
            self.sign * other.sign,
            Arc::new(divide(&mut mag, &other.magnitude)),
            true,
        ))
    }
    fn quick_pow2_check(&self) -> bool {
        self.sign > 0 && *self.get_bit_count() == 1usize
    }
    pub fn shift_right(&self, n: i32) -> BigInteger {
        if n == 0 {
            return self.clone();
        }
        if n < 0 {
            return self.shift_left(-n);
        }
        if n as usize >= *self.bit_length() {
            if self.sign < 0 {
                return (*ONE).negate();
            } else {
                return (*ZERO).clone();
            }
        }
        let result_length = (self.bit_length() - (n as usize) + 31) >> 5;
        let mut res = vec![0u32; result_length];
        let num_ints = n >> 5;
        let num_bits = n & 31;
        if num_bits == 0 {
            res.copy_from_slice(&self.magnitude[0..result_length]);
        } else {
            let num_bits2 = 32 - num_bits;
            let mut mag_pos = (self.magnitude.len() - 1 - (num_ints as usize)) as isize;
            for i in (0..result_length).rev() {
                res[i] = self.magnitude[mag_pos as usize] >> num_bits;
                mag_pos -= 1;
                if mag_pos >= 0 {
                    res[i] |= self.magnitude[mag_pos as usize] << num_bits2;
                }
            }
        }
        debug_assert!(res[0] != 0);
        BigInteger::with_check_mag(self.sign, Arc::new(res), false)
    }
    pub fn divide_and_remainder(&self, other: &BigInteger) -> (BigInteger, BigInteger) {
        if other.sign == 0 {
            panic!("divide by zero");
        }
        if self.sign == 0 {
            return ((*ZERO).clone(), (*ZERO).clone());
        } else if other.quick_pow2_check() {
            let e = other.abs().bit_length() - 1;
            let quotient = self.abs().shift_right(e as i32);

            let divide = if self.sign == other.sign {
                quotient
            } else {
                quotient.negate()
            };
            let remainder =
                BigInteger::with_check_mag(self.sign, Arc::new(self.last_n_bits(e)), true);
            return (divide, remainder);
        } else {
            let mut remainder = self.magnitude.as_ref().clone();
            let quotient = divide(&mut remainder, &other.magnitude);
            return (
                BigInteger::with_check_mag(self.sign * other.sign, Arc::new(quotient), true),
                BigInteger::with_check_mag(self.sign, Arc::new(remainder), true),
            );
        }
    }
    fn last_n_bits(&self, n: usize) -> Vec<u32> {
        if n == 0 {
            return C_ZERO_MAGNITUDE;
        }
        let num_words = (n + BITS_PER_INT - 1) / BITS_PER_INT;
        let mut result = vec![0u32; num_words];
        result.copy_from_slice(&self.magnitude[(&self.magnitude.len() - num_words)..]);
        let excess_bits = (num_words << 5) - n;
        if excess_bits > 0 {
            result[0] &= u32::MAX >> excess_bits;
        }
        result
    }

    pub fn gcd(&self, other: &BigInteger) -> Result<BigInteger> {
        if other.sign == 0 {
            return Ok(self.abs());
        }
        if self.sign == 0 {
            return Ok(other.abs());
        }

        let mut r: BigInteger;
        let mut u = self.clone();
        let mut v = other.clone();

        while v.sign != 0 {
            r = u.r#mod(&v)?;
            u = v;
            v = r;
        }
        return Ok(u);
    }

    pub fn r#mod(&self, modulus: &BigInteger) -> Result<BigInteger> {
        anyhow::ensure!(
            modulus.sign != 0,
            BcError::invalid_argument("divide by zero", "modulus")
        );
        let biggie = self.remainder(modulus)?;
        if biggie.sign >= 0 {
            Ok(biggie)
        } else {
            Ok(biggie.add(modulus))
        }
    }

    pub fn remainder(&self, division: &BigInteger) -> Result<BigInteger> {
        anyhow::ensure!(
            division.sign != 0,
            BcError::invalid_argument("divide by zero", "division")
        );
        if self.sign == 0 {
            return Ok((*ZERO).clone());
        }
        // For small values, use fast remainder method
        if division.magnitude.len() == 1 {
            let val = division.magnitude[0];
            if val > 0 {
                if val == 1 {
                    return Ok((*ZERO).clone());
                }
                let rem = self.remainder_with_u32(val);
                return if rem == 0 {
                    Ok((*ZERO).clone())
                } else {
                    Ok(BigInteger::with_check_mag(
                        self.sign,
                        Arc::new(vec![rem]),
                        false,
                    ))
                };
            }
        }
        if compare_no_leading_zeros(&self.magnitude, &division.magnitude) < 0 {
            return Ok(self.clone());
        }

        let mut result: Vec<u32>;
        if division.quick_pow2_check() {
            result = self.last_n_bits(division.abs().bit_length() - 1);
        } else {
            result = self.magnitude.as_ref().clone();
            remainder(&mut result, division.magnitude.as_ref());
        }
        Ok(BigInteger::with_check_mag(
            self.sign,
            Arc::new(result),
            true,
        ))
    }
    fn remainder_with_u32(&self, m: u32) -> u32 {
        debug_assert!(m > 0);
        let mut acc = 0u64;
        for i in 0..self.magnitude.len() {
            let pos_val = self.magnitude[i];
            acc = (acc << 32 | pos_val as u64) % m as u64;
        }
        acc as u32
    }
    pub fn i32_value(&self) -> i32 {
        if self.sign == 0 {
            return 0;
        }
        let n = self.magnitude.len();
        let v = self.magnitude[n - 1] as i32;
        if self.sign < 0 {
            v.wrapping_neg()
        } else {
            v
        }
    }

    pub fn try_get_i32_value(&self) -> Option<i32> {
        if *self.bit_length() > 31 {
            return None;
        }
        Some(self.i32_value())
    }

    pub fn get_i64_value(&self) -> i64 {
        if self.sign == 0 {
            return 0;
        }
        let n = self.magnitude.len();
        let mut v = (self.magnitude[n - 1] as i64) & IMASK;
        if n > 1 {
            v |= ((self.magnitude[n - 2] as i64) & IMASK) << 32;
        }
        if self.sign < 0 {
            v.wrapping_neg()
        } else {
            v
        }
    }

    pub fn try_get_i64_value(&self) -> Option<i64> {
        if *self.bit_length() > 63 {
            return None;
        }
        Some(self.get_i64_value())
    }

    pub fn to_string(&self) -> String {
        self.to_string_with_radix(10).unwrap()
    }

    pub fn to_string_with_radix(&self, radix: u32) -> Result<String> {
        match radix {
            2 => {}
            8 => {}
            10 => {}
            16 => {}
            _ => {
                anyhow::bail!(BcError::invalid_argument("Invalid radix", "radix"));
            }
        }

        if self.sign == 0 {
            return Ok("0".to_string());
        }

        let mut first_non_zero = 0;
        while first_non_zero < self.magnitude.len() {
            if self.magnitude[first_non_zero] != 0 {
                break;
            }
            first_non_zero += 1;
        }

        if first_non_zero == self.magnitude.len() {
            return Ok("0".to_string());
        }

        let mut sb = String::new();
        if self.sign < 0 {
            sb.push('-');
        }

        match radix {
            2 => {
                let mut pos = first_non_zero;
                sb.push_str(&format!("{:b}", self.magnitude[pos]));
                while {
                    pos += 1;
                    pos
                } < self.magnitude.len()
                {
                    append_zero_extended_string(&mut sb, &format!("{:b}", self.magnitude[pos]), 32);
                }
            }
            8 => {
                let mask = (1 << 30) - 1;
                let mut u = self.abs();
                let mut bits = *u.bit_length();
                let mut s: Vec<String> = Vec::new();
                while bits > 30 {
                    s.push(format!("{:o}", u.i32_value() & mask));
                    u = u.shift_right(30);
                    bits -= 30;
                }
                sb.push_str(&format!("{:o}", u.i32_value()));
                for i in (0..s.len()).rev() {
                    append_zero_extended_string(&mut sb, &s[i], 10);
                }
            }
            16 => {
                let mut pos = first_non_zero;
                sb.push_str(&format!("{:x}", self.magnitude[pos]));
                while {
                    pos += 1;
                    pos
                } < self.magnitude.len()
                {
                    append_zero_extended_string(&mut sb, &format!("{:x}", self.magnitude[pos]), 8);
                }
            }
            // TODO This could work for other radices if there is an alternative to Convert.ToString method
            10 => {
                let q = self.abs();
                if *q.bit_length() < 64 {
                    sb.push_str(&format!("{}", q.get_i64_value()));
                } else {
                    // TODO Could cache the moduli for each radix (soft reference?)
                    let mut moduli: Vec<BigInteger> = Vec::new();
                    let mut r = BigInteger::with_u32(radix);
                    while r <= q {
                        moduli.push(r.clone());
                        r = r.square();
                    }

                    let scale = moduli.len();
                    sb.reserve(sb.len() + (1 << scale));

                    to_string_with_moduli(&mut sb, radix, &moduli, scale, &q);
                }
            }
            _ => {}
        }
        Ok(sb)
    }

    pub fn to_long_value(&self) -> i64 {
        if self.sign == 0 {
            return 0;
        }
        let n = self.magnitude.len();
        let mut v = self.magnitude[n - 1] as i64 & IMASK;
        if n > 1 {
            v |= (self.magnitude[n - 2] as i64 & IMASK) << 32;
        }
        if self.sign < 0 {
            -v
        } else {
            v
        }
    }

    pub fn square(&self) -> BigInteger {
        if self.sign == 0 {
            return (*ZERO).clone();
        }
        if self.quick_pow2_check() {
            return self.shift_left((self.abs().bit_length() - 1) as i32);
        }
        let mut res_length = self.magnitude.len() << 1;
        if self.magnitude[0] >> 16 == 0 {
            res_length -= 1;
        }
        let mut res = vec![0u32; res_length];
        square(&mut res, &self.magnitude);
        BigInteger::with_check_mag(1, Arc::new(res), false)
    }

    pub fn shift_left(&self, n: i32) -> BigInteger {
        if self.sign == 0 || self.magnitude.len() == 0 {
            return (*ZERO).clone();
        }
        if n == 0 {
            return self.clone();
        }
        if n < 0 {
            return self.shift_right(-n);
        }

        let result = BigInteger::with_check_mag(
            self.sign,
            Arc::new(shift_left(&self.magnitude, n as usize)),
            true,
        );

        result.bits.get_or_init(|| {
            if result.sign > 0 {
                *self.get_bit_count()
            } else {
                *self.get_bit_count() + n as usize
            }
        });

        result
            .bit_length
            .get_or_init(|| self.bit_length() + n as usize);

        result
    }

    pub fn test_bit(&self, n: usize) -> bool {
        if self.sign < 0 {
            return !self.not().test_bit(n);
        }

        let word_num = n / 32;
        if word_num >= self.magnitude.len() {
            return false;
        }
        let word = self.magnitude[self.magnitude.len() - 1 - word_num];
        return ((word >> (n % 32)) & 1) != 0;
    }

    fn check_probable_prime(
        &self,
        certainty: i32,
        random: &mut dyn RandomSource,
        randomly_selected: bool,
    ) -> Result<bool> {
        debug_assert!(certainty > 0);
        debug_assert!(self > &(*TWO));
        debug_assert!(self.test_bit(0));

        // Try to reduce the penalty for really small numbers
        let num_lists = std::cmp::min(self.bit_length() - 1, PRIME_LISTS.len());
        for i in 0..num_lists {
            let test = self.remainder_with_u32(*(&(*PRIME_PRODUCTS)[i]));
            let prime_list = &(*PRIME_LISTS)[i];
            for j in 0..prime_list.len() {
                let prime = prime_list[j];
                let q_rem = test % prime;
                if q_rem == 0 {
                    return Ok(*self.bit_length() < 16 && self.i32_value() as u32 == prime);
                }
            }
        }
        // TODO Special case for < 10^16 (RabinMiller fixed list)
        // if self.get_bit_length() < 30 {
        //     RabinMiller against 2, 3, 5, 7, 11, 13, 23 is sufficient
        // }

        // TODO Is it worth trying to create a hybrid of these two?
        return self.rabin_miller_test_with_randomly_selected(certainty, random, randomly_selected);

        // // self.solovay_strassen_test(certainty, random);
        // let rb_test = self.rabin_miller_test(certainty, random);
        // let ss_test = solovay_strassen_test(certainty, random);

        // debug_assert!(rb_test == ss_test);

        // return rb_test;
    }

    pub fn rabin_miller_test(&self, certainty: i32, random: &mut dyn RandomSource) -> Result<bool> {
        return self.rabin_miller_test_with_randomly_selected(certainty, random, false);
    }

    pub(crate) fn rabin_miller_test_with_randomly_selected(
        &self,
        certainty: i32,
        random: &mut dyn RandomSource,
        randomly_selected: bool,
    ) -> Result<bool> {
        let bits = *self.bit_length();

        debug_assert!(certainty > 0);
        debug_assert!(bits > 2);
        debug_assert!(self.test_bit(0));

        let mut iterations = ((certainty - 1) / 2) + 1;
        if randomly_selected {
            let iters_for_100_cert = if bits >= 1024 {
                4
            } else if bits >= 512 {
                8
            } else if bits >= 256 {
                16
            } else {
                50
            };
            if certainty < 100 {
                iterations = std::cmp::min(iters_for_100_cert, iterations);
            } else {
                iterations -= 50;
                iterations += iters_for_100_cert;
            }
        }

        // let n = 1 + d . 2^s
        let n = self.clone();
        let s = n.get_lowest_set_bit_mask_first(u32::MAX << 1);
        debug_assert!(s >= 1);
        let r = n.shift_right(s as i32);

        // NOTE: Avoid conversion to/from Montgomery form and check for R/-R as result instead

        let mont_radix = (*ONE)
            .shift_left((32 * n.magnitude.len()) as i32)
            .remainder(&n)?;
        let minus_mont_radix = n.subtract(&mont_radix);

        let mut y_accum = vec![0u32; n.magnitude.len() + 1];

        loop {
            let mut a: BigInteger;
            loop {
                a = BigInteger::with_random(*n.bit_length(), random);
                //a = BigInteger::with_string("4571627816229255254").expect("msg");
                if a.sign == 0
                    || a >= n
                    || is_equal_magnitude(&a.magnitude, &mont_radix.magnitude)
                    || is_equal_magnitude(&a.magnitude, &minus_mont_radix.magnitude)
                {
                    // Nothing
                } else {
                    break;
                }
            }

            let mut y = Self::mod_pow_monty(&mut y_accum, &a, &r, &n, false)?;

            if y != mont_radix {
                let mut j = 0;
                while y != minus_mont_radix {
                    j += 1;
                    if j == s {
                        return Ok(false);
                    }
                    y = Self::mod_square_monty(&mut y_accum, &y, &n);

                    if y == mont_radix {
                        return Ok(false);
                    }
                }
            }
            iterations -= 1;
            if iterations > 0 {
                // Nothing
            } else {
                break;
            }
        }
        return Ok(true);
    }

    fn get_lowest_set_bit_mask_first(&self, first_word_mask_x: u32) -> i32 {
        let mut w = self.magnitude.len();
        let mut offset = 0i32;
        w -= 1;
        let mut word = self.magnitude[w] & first_word_mask_x;
        debug_assert!(self.magnitude[0] != 0);
        while word == 0 {
            w -= 1;
            word = self.magnitude[w];
            offset += 32;
        }

        offset += word.trailing_zeros() as i32;
        return offset;
    }

    fn mod_pow_monty(
        y_accm: &mut [u32],
        b: &BigInteger,
        e: &BigInteger,
        m: &BigInteger,
        convert: bool,
    ) -> Result<BigInteger> {
        let n = m.magnitude.len();
        let pow_r = 32 * n;
        let small_monty_modulus = m.bit_length() + 2 <= pow_r;
        let m_dash = m.get_m_quote();

        // tmp = this * R mod m
        let mut b1 = b.clone();
        if convert {
            b1 = b1.shift_left(pow_r as i32).remainder(m)?;
        }
        debug_assert!(y_accm.len() == n + 1);

        let mut z_val = b1.magnitude.to_vec();
        debug_assert!(z_val.len() <= n);
        if z_val.len() < n {
            let mut tmp = vec![0u32; n];
            tmp[(n - z_val.len())..].copy_from_slice(&z_val);
            z_val = tmp;
        }

        // Sliding window from MSW to LSW

        let mut extra_bits = 0;

        // Filter the common case of small RSA exponents with few bits set
        if e.magnitude.len() > 1 || *e.get_bit_count() > 2 {
            let exp_length = *e.bit_length();
            while exp_length > EXP_WINDOW_THRESHOLDS[extra_bits] {
                extra_bits += 1;
            }
        }

        let num_powers = 1usize << extra_bits;
        let mut odd_powers = vec![vec![0u32; 0]; num_powers];
        odd_powers[0] = z_val.clone();

        let mut z_squared = z_val.clone(); // todo!()
        square_monty(
            y_accm,
            &mut z_squared,
            &m.magnitude,
            m_dash,
            small_monty_modulus,
        );

        for i in 1..num_powers {
            odd_powers[i] = odd_powers[i - 1].clone();
            multiply_monty(
                y_accm,
                &mut odd_powers[i],
                &z_squared,
                &m.magnitude,
                m_dash,
                small_monty_modulus,
            );
        }

        let window_list = get_window_list(&e.magnitude, extra_bits);
        debug_assert!(window_list.len() > 1);

        let mut window = window_list[0];
        let mut mult = window & 0xFF;
        let mut last_zeros = window >> 8;

        let mut y_val: Vec<u32>;
        if mult == 1 {
            y_val = z_squared;
            last_zeros = last_zeros.wrapping_sub(1);
        } else {
            y_val = odd_powers[(mult >> 1) as usize].clone();
        }
        let mut window_pos = 1;
        while {
            window = window_list[window_pos];
            window_pos += 1;
            window
        } != u32::MAX
        {
            mult = window & 0xFF;
            let bits = last_zeros as i32 + bit_len(mult) as i32;
            for _ in 0..bits {
                square_monty(
                    y_accm,
                    &mut y_val,
                    &m.magnitude,
                    m_dash,
                    small_monty_modulus,
                );
            }

            multiply_monty(
                y_accm,
                &mut y_val,
                &odd_powers[(mult >> 1) as usize],
                &m.magnitude,
                m_dash,
                small_monty_modulus,
            );

            last_zeros = window >> 8;
        }

        for _ in 0..last_zeros {
            square_monty(
                y_accm,
                &mut y_val,
                &m.magnitude,
                m_dash,
                small_monty_modulus,
            );
        }

        if convert {
            // Return y * R^(-1) mod m
            montgomery_reduce(&mut y_val, &m.magnitude, m_dash);
        } else if small_monty_modulus && compare_to(&y_val, &m.magnitude) >= 0 {
            subtract(&mut y_val, &m.magnitude);
        }

        Ok(BigInteger::with_check_mag(1, Arc::new(y_val), true))
    }

    fn mod_square_monty(y_accum: &mut [u32], b: &BigInteger, m: &BigInteger) -> BigInteger {
        let n = m.magnitude.len();
        let pow_r = 32 * n;
        let small_monty_modulus = m.bit_length() + 2 <= pow_r;
        let m_dash = m.get_m_quote();

        debug_assert!(y_accum.len() == n + 1);

        let z_val = b.magnitude.to_vec();
        debug_assert!(z_val.len() <= n);

        let mut y_val = vec![0u32; n];
        y_val[n - z_val.len()..].copy_from_slice(&z_val);

        square_monty(
            y_accum,
            &mut y_val,
            &m.magnitude,
            m_dash,
            small_monty_modulus,
        );

        if small_monty_modulus && compare_to(&y_val, &m.magnitude) >= 0 {
            subtract(&mut y_val, &m.magnitude);
        }

        BigInteger::with_check_mag(1, Arc::new(y_val), true)
    }

    /// Calculate mQuote = -m^(-1) mod b with b = 2^32 (32 = word size)
    fn get_m_quote(&self) -> u32 {
        debug_assert!(self.sign > 0);
        let d = 0u32.wrapping_sub(self.magnitude[self.magnitude.len() - 1]);
        debug_assert!((d & 1) != 0);
        inverse_u32(d)
    }

    pub fn set_bit(&self, n: usize) -> BigInteger {
        if self.test_bit(n) {
            return self.clone();
        }
        // TODO Handle negative values and zero
        if self.sign > 0 && n < (*self.bit_length() - 1) {
            return self.flip_existing_bit(n);
        } else {
            return self.or(&(*ONE).shift_left(n as i32));
        }
    }

    fn flip_existing_bit(&self, n: usize) -> BigInteger {
        debug_assert!(self.sign > 0);
        debug_assert!(n < self.bit_length() - 1);

        let mut mag = self.magnitude.to_vec();
        let mag_len = mag.len();
        let v = (1 << (n as i32 & 31)) as u32;
        mag[mag_len - 1 - (n as i32 >> 5) as usize] ^= v;
        BigInteger::with_check_mag(self.sign, Arc::new(mag), false)
    }

    pub fn clear_bit(&self, n: usize) -> BigInteger {
        if !self.test_bit(n) {
            return self.clone();
        }

        // TODO Handle negative values
        if self.sign > 0 && n < (self.bit_length() - 1) {
            return self.flip_existing_bit(n);
        }

        let r1 = (*ONE).shift_left(n as i32);
        let r2 = self.and_not(&r1);
        r2
    }

    pub fn multiply(&self, other: &BigInteger) -> BigInteger {
        if self == other {
            return self.square();
        }

        if (self.sign & other.sign) == 0 {
            return (*ZERO).clone();
        }

        if other.quick_pow2_check() {
            let result = self.shift_left((other.abs().bit_length() - 1) as i32);
            return if other.sign > 0 {
                result
            } else {
                result.negate()
            };
        }

        if self.quick_pow2_check() {
            let result = other.shift_left((self.abs().bit_length() - 1) as i32);
            return if self.sign > 0 {
                result
            } else {
                result.negate()
            };
        }

        let res_length = self.magnitude.len() + other.magnitude.len();
        let mut res = vec![0u32; res_length];

        multiply(&mut res, &self.magnitude, &other.magnitude);

        let res_sign = self.sign ^ other.sign ^ 1;

        BigInteger::with_check_mag(res_sign, Arc::new(res), true)
    }

    pub fn flip_bit(&self, n: usize) -> BigInteger {
        // TODO Handle negative values and zero
        if self.sign > 0 && n < (*self.bit_length() - 1) {
            return self.flip_existing_bit(n);
        }
        let n1 = &(*ONE).shift_left(n as i32);
        return self.xor(n1);
    }

    pub fn xor(&self, value: &BigInteger) -> BigInteger {
        if self.sign == 0 {
            return value.clone();
        }
        if value.sign == 0 {
            return self.clone();
        }

        let a_mag = if self.sign > 0 {
            self.magnitude.clone()
        } else {
            self.add(&(*ONE)).magnitude.clone()
        };
        let b_mag = if value.sign > 0 {
            value.magnitude.clone()
        } else {
            value.add(&(*ONE)).magnitude.clone()
        };

        // TODO Can just replace with sign != value.sign?
        let result_neg = (self.sign < 0 && value.sign >= 0) || (self.sign >= 0 && value.sign < 0);
        let result_length = std::cmp::max(a_mag.len(), b_mag.len());
        let mut result_mag = vec![0u32; result_length];

        let a_start = result_mag.len() - a_mag.len();
        let b_start = result_mag.len() - b_mag.len();

        for i in 0..result_mag.len() {
            let mut a_word = if i >= a_start {
                a_mag[i - a_start]
            } else {
                0u32
            };
            let mut b_word = if i >= b_start {
                b_mag[i - b_start]
            } else {
                0u32
            };

            if self.sign < 0 {
                a_word = !a_word;
            }
            if value.sign < 0 {
                b_word = !b_word;
            }

            result_mag[i] = a_word ^ b_word;

            if result_neg {
                result_mag[i] = !result_mag[i];
            }
        }

        // TODO Optimise this case
        let mut result = BigInteger::with_check_mag(1, Arc::new(result_mag), true);
        if result_neg {
            result = result.not();
        }
        result
    }

    pub fn pow(&self, mut exp: u32) -> Result<BigInteger> {
        if exp == 0 {
            return Ok((*ONE).clone());
        }
        if self.sign == 0 {
            return Ok(self.clone());
        }
        if self.quick_pow2_check() {
            let pow_of_2 = exp as u64 * (self.bit_length() - 1) as u64;
            anyhow::ensure!(
                pow_of_2 <= i32::MAX as u64,
                BcError::arithmetic_error("Result too large")
            );
            return Ok((*ONE).shift_left(pow_of_2 as i32));
        }
        let mut y = (*ONE).clone();
        let mut z = self.clone();
        loop {
            if (exp & 0x1) == 1 {
                y = y.multiply(&z);
            }
            exp >>= 1;
            if exp == 0 {
                break;
            }
            z = z.multiply(&z);
        }
        Ok(y)
    }

    pub fn get_sign_value(&self) -> i32 {
        self.sign
    }

    pub fn to_vec(&self) -> Vec<u8> {
        self.to_vec_with_signed(false)
    }

    pub fn to_vec_unsigned(&self) -> Vec<u8> {
        self.to_vec_with_signed(true)
    }

    fn to_vec_with_signed(&self, unsigned: bool) -> Vec<u8> {
        if self.sign == 0 {
            return if unsigned { vec![0u8; 0] } else { vec![0u8; 1] };
        }
        let n_bits = if unsigned && self.sign > 0 {
            *self.bit_length()
        } else {
            *self.bit_length() + 1
        };
        let n_bytes = get_bytes_length(n_bits);

        let mut bytes = vec![0u8; n_bytes];
        let mut mag_index = self.magnitude.len();
        let mut bytes_index = bytes.len();

        if self.sign > 0 {
            while mag_index > 1 {
                let mag = self.magnitude[{
                    mag_index -= 1;
                    mag_index
                }];
                bytes_index -= 4;
                u32_to_be_bytes(mag, &mut bytes[bytes_index..]);
            }

            let mut last_mag = self.magnitude[0];
            while last_mag > u8::MAX as u32 {
                bytes[{
                    bytes_index -= 1;
                    bytes_index
                }] = (last_mag & 0xFF) as u8;
                last_mag >>= 8;
            }

            bytes[{
                bytes_index -= 1;
                bytes_index
            }] = last_mag as u8;
            debug_assert!(bytes_index & 0x1 == bytes_index);
        } else {
            let mut carry = true;
            while mag_index > 1 {
                let mut mag = !self.magnitude[{
                    mag_index -= 1;
                    mag_index
                }];
                if carry {
                    carry = {
                        mag = mag.wrapping_add(1);
                        mag
                    } == u32::MIN;
                }
                bytes_index -= 4;
                u32_to_be_bytes(mag, &mut bytes[bytes_index..]);
            }

            let mut last_mag = self.magnitude[0];
            if carry {
                // Never wraps because magnitude[0] != 0
                last_mag -= 1;
            }

            while last_mag > u8::MAX as u32 {
                bytes[{
                    bytes_index -= 1;
                    bytes_index
                }] = (!last_mag) as u8;
                last_mag >>= 8;
            }

            bytes[{
                bytes_index -= 1;
                bytes_index
            }] = (!last_mag) as u8;
            debug_assert!(bytes_index & 0x1 == bytes_index);
            if bytes_index != 0 {
                bytes[{
                    bytes_index -= 1;
                    bytes_index
                }] = u8::MAX;
            }
        }
        bytes
    }

    pub fn next_probable_prime(&self) -> Result<BigInteger> {
        anyhow::ensure!(
            self.sign >= 0,
            BcError::arithmetic_error("Negative numbers cannot be prime")
        );
        if self < &(*TWO) {
            return Ok((*TWO).clone());
        }

        let mut n = self.inc().set_bit(0);
        while !n.check_probable_prime(100, &mut DefaultRandomSource::default(), false)? {
            n = n.add(&(*TWO));
        }
        Ok(n)
    }

    pub fn is_probable_prime(&self, certainty: i32) -> Result<bool> {
        self.is_probable_prime_with_randomly_selected(certainty, false)
    }

    pub(crate) fn is_probable_prime_with_randomly_selected(
        &self,
        certainty: i32,
        randomly_selected: bool,
    ) -> Result<bool> {
        if certainty <= 0 {
            return Ok(true);
        }
        let n = self.abs();

        if !n.test_bit(0) {
            return Ok(n == (*TWO));
        }

        if n == (*ONE) {
            return Ok(false);
        }

        Ok(n.check_probable_prime(
            certainty,
            &mut DefaultRandomSource::default(),
            randomly_selected,
        )?)
    }

    pub fn mod_pow(&self, e: &BigInteger, m: &BigInteger) -> Result<BigInteger> {
        anyhow::ensure!(
            m.sign > 0,
            BcError::arithmetic_error("Modulus must be positive")
        );
        if m == &(*ONE) {
            return Ok((*ZERO).clone());
        }

        if e.sign == 0 {
            return Ok((*ONE).clone());
        }

        if self.sign == 0 {
            return Ok((*ZERO).clone());
        }

        let neg_exp = e.sign < 0;
        let mut e1 = e.clone();
        if neg_exp {
            e1 = e.negate();
        }

        let mut result = self.r#mod(m)?;
        if &e1 != &(*ONE) {
            if m.magnitude[m.magnitude.len() - 1] & 1 == 0 {
                result = Self::mod_pow_barrett(&result, &e1, m)?;
            } else {
                let mut y_accum = vec![0u32; m.magnitude.len() + 1];
                result = Self::mod_pow_monty(&mut y_accum, &result, &e1, m, true)?;
            }
        }

        if neg_exp {
            result = result.mod_inverse(m)?;
        }
        Ok(result)
    }

    fn mod_pow_barrett(b: &BigInteger, e: &BigInteger, m: &BigInteger) -> Result<BigInteger> {
        let k = m.magnitude.len();
        let mr = (*ONE).shift_left(((k + 1) << 5) as i32);
        let yu = (*ONE).shift_left((k << 6) as i32).divide(m)?;

        // Sliding window from MSW to LSW
        let mut extra_bits = 0;
        let exp_length = *e.bit_length();
        while exp_length > EXP_WINDOW_THRESHOLDS[extra_bits] {
            extra_bits += 1;
        }

        let num_powers = 1usize << extra_bits;
        let mut odd_powers = vec![(*ZERO).clone(); num_powers];
        odd_powers[0] = b.clone();

        let b2 = Self::reduce_barrett(&b.square(), m, &mr, &yu);

        for i in 1..num_powers {
            odd_powers[i] = Self::reduce_barrett(&(odd_powers[i - 1].multiply(&b2)), m, &mr, &yu);
        }

        let window_list = get_window_list(&e.magnitude, extra_bits);
        debug_assert!(window_list.len() > 1);

        let mut window = window_list[0];
        let mut mult = window & 0xFF;
        let mut last_zeros = window >> 8;

        let mut y: BigInteger;
        if mult == 1 {
            y = b2.clone();
            last_zeros -= 1;
        } else {
            y = odd_powers[(mult >> 1) as usize].clone();
        }

        let mut window_pos = 1;
        while {
            window = window_list[window_pos];
            window_pos += 1;
            window
        } != u32::MAX
        {
            mult = window & 0xFF;
            let bits = last_zeros + bit_len(mult);
            for _ in 0..bits {
                y = Self::reduce_barrett(&y.square(), m, &mr, &yu);
            }
            y = Self::reduce_barrett(
                &(y.multiply(&odd_powers[(mult >> 1) as usize])),
                m,
                &mr,
                &yu,
            );
            last_zeros = window >> 8;
        }

        for _ in 0..last_zeros {
            y = Self::reduce_barrett(&(y.square()), m, &mr, &yu);
        }

        Ok(y)
    }

    fn reduce_barrett(
        x: &BigInteger,
        m: &BigInteger,
        mr: &BigInteger,
        yu: &BigInteger,
    ) -> BigInteger {
        let x_len = *x.bit_length();
        let m_len = *m.bit_length();
        if x_len < m_len {
            return x.clone();
        }

        let mut x1 = x.clone();

        if x_len - m_len > 1 {
            let k = m.magnitude.len();

            let q1 = x.divide_words((k - 1) as u32);
            let q2 = q1.multiply(yu);
            let q3 = q2.divide_words((k + 1) as u32);

            let r1 = x.remainder_words((k + 1) as u32);
            let r2 = q3.multiply(m);
            let r3 = r2.remainder_words((k + 1) as u32);

            x1 = r1.subtract(&r3);
            if x.sign < 0 {
                x1 = x1.add(mr);
            }
        }

        while &x1 >= m {
            x1 = x1.subtract(m);
        }

        x1
    }

    fn divide_words(&self, w: u32) -> BigInteger {
        let n = self.magnitude.len();
        if w as usize >= n {
            return (*ZERO).clone();
        }
        let mut mag = vec![0u32; n - w as usize];
        mag.copy_from_slice(&self.magnitude[..(n - w as usize)]);
        BigInteger::with_check_mag(self.sign, Arc::new(mag), false)
    }

    fn remainder_words(&self, w: u32) -> BigInteger {
        let n = self.magnitude.len();
        if w as usize >= n {
            return self.clone();
        }
        let mut mag = vec![0u32; w as usize];
        mag.copy_from_slice(&self.magnitude[(n - w as usize)..]);
        BigInteger::with_check_mag(self.sign, Arc::new(mag), false)
    }

    pub fn mod_inverse(&self, modulus: &BigInteger) -> Result<BigInteger> {
        anyhow::ensure!(
            modulus.sign > 0,
            BcError::arithmetic_error("Modulus must be positive")
        );
    
        if modulus.quick_pow2_check() {
            return Ok(self.mod_inverse_pow2(modulus)?);
        }

        let d = self.remainder(modulus)?;
        let (gcd, mut x) = ext_euclid(&d, modulus);
        anyhow::ensure!(
            gcd == (*ONE),
            BcError::arithmetic_error("Numbers not relatively prime")
        );
        
        if x.sign < 0 {
            x = x.add(modulus);
        }
        Ok(x)
    }

    fn mod_inverse_pow2(&self, m: &BigInteger) -> Result<BigInteger> {
        debug_assert!(m.sign > 0);
        debug_assert!(*m.get_bit_count() == 1);

        anyhow::ensure!(
            self.test_bit(0),
            BcError::arithmetic_error("Numbers not relatively prime")
        );
        
        let pow = *m.bit_length() << 1;
        let mut inv64 = inverse_u64(self.get_i64_value() as u64) as i64;
        if pow < 64 {
            inv64 &= (1 << pow) - 1;
        }

        let mut x = BigInteger::with_i64(inv64 as i64);

        if pow > 64 {
            let d = self.remainder(m)?;
            let mut bits_correct = 64;

            loop {
                let t = x.multiply(&d).remainder(m)?;
                x = x.multiply(&(*TWO).subtract(&t)).remainder(m)?;
                bits_correct <<= 1;

                if bits_correct < pow {
                    // nothing
                } else {
                    break;
                }
            }
        }

        if x.sign < 0 {
            x = x.add(m);
        }
        Ok(x)
    }

    pub fn get_lowest_set_bit(&self) -> i32 {
        if self.sign == 0 {
            return -1;
        }
        self.get_lowest_set_bit_mask_first(u32::MAX)
    }

    pub fn max(&self, value: &BigInteger) -> BigInteger {
        if self < value {
            value.clone()
        } else {
            self.clone()
        }
    }

    pub fn min(&self, value: &BigInteger) -> BigInteger {
        if self < value {
            self.clone()
        } else {
            value.clone()
        }
    }
}
impl PartialEq for BigInteger {
    fn eq(&self, other: &Self) -> bool {
        return self.sign == other.sign
            && is_equal_magnitude(self.magnitude.as_slice(), other.magnitude.as_slice());
    }
}

impl PartialOrd for BigInteger {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.sign < other.sign {
            return Some(std::cmp::Ordering::Less);
        }
        if self.sign > other.sign {
            return Some(std::cmp::Ordering::Greater);
        }
        if self.sign == 0 {
            return Some(std::cmp::Ordering::Equal);
        }
        let compare = self.sign
            * compare_no_leading_zeros(self.magnitude.as_slice(), other.magnitude.as_slice());
        if compare == 0 {
            return Some(std::cmp::Ordering::Equal);
        }
        if compare < 0 {
            return Some(std::cmp::Ordering::Less);
        }
        if compare > 0 {
            return Some(std::cmp::Ordering::Greater);
        }
        None
    }
}

impl Hash for BigInteger {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.magnitude.as_ref().hash(state);
        self.sign.hash(state);
    }
}

impl Display for BigInteger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl Debug for BigInteger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

fn init_be(buffer: &[u8]) -> (Vec<u32>, i32) {
    if (buffer[0] as i8) >= 0 {
        let magnitude = make_magnitude_be(buffer);
        let sign = if magnitude.is_empty() { 0 } else { 1 };
        (magnitude, sign)
    } else {
        let magnitude = make_magnitude_be_negative(buffer);
        let sign = -1;
        (magnitude, sign)
    }
}

fn init_le(buffer: &[u8]) -> (Vec<u32>, i32) {
    if (buffer[buffer.len() - 1] as i8) >= 0 {
        let magnitude = make_magnitude_le(buffer);
        let sign = if magnitude.is_empty() { 0 } else { 1 };
        return (magnitude, sign);
    } else {
        let magnitude = make_magnitude_le_negative(buffer);
        let sign = -1;
        return (magnitude, sign);
    }
}

// strip leading zeros
fn strip_prefix_value<T: PartialEq>(buffer: &[T], v: T) -> &[T] {
    let mut find = false;
    let mut pos: usize = 0usize;
    for i in 0..buffer.len() {
        if buffer[i] != v {
            pos = i;
            find = true;
            break;
        }
    }
    if find {
        &buffer[pos..]
    } else {
        &buffer[pos..0]
    }
}

fn strip_suffix_value(buffer: &[u8], v: u8) -> &[u8] {
    let mut find = false;
    let mut pos = 0usize;
    for i in (0..buffer.len()).rev() {
        if buffer[i] != v {
            pos = i + 1;
            find = true;
            break;
        }
    }
    if find {
        &buffer[..pos]
    } else {
        &buffer[pos..pos]
    }
}

// make magnitude
fn make_magnitude_be(buffer: &[u8]) -> Vec<u32> {
    let end = buffer.len();
    // strip leading zeros
    let mut start = 0usize;
    while start < end {
        if buffer[start] != 0 {
            break;
        }
        start += 1;
    }
    let n_bytes = end - start;
    if n_bytes == 0 {
        return C_ZERO_MAGNITUDE;
    }
    let des_len = (n_bytes + C_BYTES_PRE_INT - 1) / C_BYTES_PRE_INT;
    debug_assert!(des_len > 0);
    let mut magnitude = vec![0u32; des_len];
    let first = ((n_bytes - 1) % C_BYTES_PRE_INT) + 1;
    magnitude[0] = be_to_u32_low(&buffer[start..(start + first)]);
    be_to_u32_buffer(&buffer[(start + first)..], &mut magnitude[1..]);
    magnitude
}

fn make_magnitude_be_negative(buffer: &[u8]) -> Vec<u32> {
    let sub_slice = strip_prefix_value(buffer, 0xff);
    if sub_slice.is_empty() {
        return vec![1u32];
    }
    let mut inverse = vec![0u8; sub_slice.len()];
    let mut index = 0usize;
    while index < sub_slice.len() {
        inverse[index] = !sub_slice[index];
        index += 1;
    }
    debug_assert!(index == sub_slice.len());
    while inverse[{
        index -= 1;
        index
    }] == u8::MAX
    {
        inverse[index] = u8::MIN;
    }
    inverse[index] += 1;
    make_magnitude_be(&inverse)
}

fn make_magnitude_le(buffer: &[u8]) -> Vec<u32> {
    let sub_slice = strip_suffix_value(buffer, 0x0);
    if sub_slice.is_empty() {
        return C_ZERO_MAGNITUDE;
    }
    let count = (sub_slice.len() + C_BYTES_PRE_INT) / C_BYTES_PRE_INT;
    debug_assert!(count > 0);
    let mut magnitude = vec![0u32; count];
    // 01  02  03  04 | 05  06  07  08 | 09  10  11 |
    let partial = sub_slice.len() % C_BYTES_PRE_INT;
    let mut pos = sub_slice.len() - partial;
    magnitude[0] = le_to_u32_low(&sub_slice[pos..(pos + partial)]);

    for i in 1..count {
        pos -= C_BYTES_PRE_INT;
        magnitude[i] =
            u32::from_le_bytes(sub_slice[pos..(pos + C_BYTES_PRE_INT)].try_into().unwrap());
    }
    return magnitude;
}

fn make_magnitude_le_negative(buffer: &[u8]) -> Vec<u32> {
    let mut inverse = vec![0u8; buffer.len()];
    let mut index = 0usize;
    while index < buffer.len() {
        inverse[index] = !buffer[index];
        index += 1;
    }
    debug_assert!(index == buffer.len());
    index -= 1;
    while inverse[index] == u8::MAX {
        inverse[index] = u8::MIN;
        index -= 1;
    }
    inverse[index] += 1;
    make_magnitude_le(&inverse)
}

/**
 * return a = a + b - b preserved.
 */
fn add_magnitudes(a: &mut [u32], b: &[u32]) {
    let mut ti = (a.len() - 1) as isize;
    let mut vi = (b.len() - 1) as isize;
    let mut m = 0u64;
    while vi >= 0 {
        m += a[ti as usize] as u64 + b[vi as usize] as u64;
        vi -= 1;
        a[ti as usize] = m as u32;
        ti -= 1;
        m >>= 32;
    }
    if m != 0 {
        while ti >= 0 && {
            a[ti as usize] = a[ti as usize].wrapping_add(1);
            a[ti as usize]
        } == u32::MIN
        {
            ti -= 1;
        }
    }
}

fn compare_no_leading_zeros(x: &[u32], y: &[u32]) -> i32 {
    let diff = x.len() as isize - y.len() as isize;
    if diff != 0 {
        return if diff < 0 { -1 } else { 1 };
    }

    // lengths of magnitudes the same, test the magnitude values
    let mut x_index = 0;
    let mut y_index = 0;
    while x_index < x.len() {
        let v1 = x[x_index];
        let v2 = y[y_index];
        if v1 != v2 {
            return if v1 < v2 { -1 } else { 1 };
        }
        x_index += 1;
        y_index += 1;
    }
    return 0;
}

fn do_sub_big_lil(big: &[u32], lil: &[u32]) -> Vec<u32> {
    let mut res = big.to_vec();
    subtract(res.as_mut_slice(), lil);
    res
}

// returns x = x - y - we assume x is >= y
fn subtract(x: &mut [u32], y: &[u32]) {
    debug_assert!(0 < y.len());
    debug_assert!(x.len() >= y.len());

    let mut it = x.len() as i32;
    let mut iv = y.len() as i32;
    let mut m: i64;
    let mut borrow = 0i32;
    loop {
        m = (x[{
            it -= 1;
            it as usize
        }] as i64
            & IMASK)
            - (y[{
                iv -= 1;
                iv as usize
            }] as i64
                & IMASK)
            + borrow as i64;
        x[it as usize] = m as u32;
        borrow = (m >> 63) as i32;
        if !(iv > 0) {
            break;
        }
    }
    if borrow != 0 {
        loop {
            it -= 1;
            x[it as usize] = x[it as usize].wrapping_sub(1);
            if x[it as usize] != u32::MAX {
                break;
            }
        }
    }
}

fn calc_bit_length(sign: i32, magnitude: &[u32]) -> usize {
    let mut indx = 0usize;
    loop {
        if indx >= magnitude.len() {
            return 0;
        }
        if magnitude[indx] != 0 {
            break;
        }
        indx += 1;
    }

    // bit length for everything after the first int
    let mut bit_length = 32 * ((magnitude.len() - indx) - 1);

    // and determine bitlength of first int
    let first_mag = magnitude[indx];
    bit_length += bit_len(first_mag) as usize;

    // Check for negative powers of two
    if sign < 0 && ((first_mag & (-(first_mag as i64)) as u32) == first_mag) {
        loop {
            if {
                indx += 1;
                indx
            } >= magnitude.len()
            {
                bit_length -= 1;
                break;
            }
            if magnitude[indx] == 0 {
                // nothing
            } else {
                break;
            }
        }
    }
    bit_length
}

fn bit_len(v: u32) -> u32 {
    32 - v.leading_zeros()
}

fn is_equal_magnitude(x: &[u32], y: &[u32]) -> bool {
    if x.len() != y.len() {
        return false;
    }
    for i in 0..x.len() {
        if x[i] != y[i] {
            return false;
        }
    }
    true
}

// return z = x / y - done in place (z value preserved, x contains the remainder)
fn divide(x: &mut [u32], y: &[u32]) -> Vec<u32> {
    let mut x_start = 0;
    while x_start < x.len() && x[x_start] == 0 {
        x_start += 1;
    }
    let mut y_start = 0;
    while y_start < y.len() && y[y_start] == 0 {
        y_start += 1;
    }

    debug_assert!(y_start < y.len());

    let mut xy_cmp = compare_no_leading_zeros(&x[x_start as usize..], &y[y_start as usize..]);
    let mut count: Vec<u32>;

    if xy_cmp > 0 {
        let y_bit_length = calc_bit_length(1, &y[y_start..]);
        let mut x_bit_length = calc_bit_length(1, &x[x_start..]);
        let mut shift = x_bit_length as isize - y_bit_length as isize;

        let mut icount: Vec<u32>;
        let mut i_count_start = 0;

        let mut c: Vec<u32>;
        let mut c_start = 0;
        let mut c_bit_length = y_bit_length;
        if shift > 0 {
            icount = vec![0u32; (shift as usize >> 5) + 1];
            icount[0] = 1u32 << (shift as u32 % 32);

            c = shift_left(y, shift as usize);
            c_bit_length += shift as usize;
        } else {
            icount = vec![1u32];
            let len = y.len() - y_start;

            c = vec![0u32; len];
            c.copy_from_slice(&y[y_start as usize..]);
        }
        count = vec![0u32; icount.len()];
        loop {
            if c_bit_length < x_bit_length
                || compare_no_leading_zeros(&x[x_start as usize..], &c[c_start..]) >= 0
            {
                subtract(&mut x[x_start as usize..], &c[c_start..]);
                add_magnitudes(&mut count, &icount);
                while x[x_start as usize] == 0 {
                    x_start += 1;
                    if x_start == x.len() {
                        return count;
                    }
                }
                x_bit_length = 32 * (x.len() - x_start - 1) + bit_len(x[x_start]) as usize;
                if x_bit_length <= y_bit_length {
                    if x_bit_length < y_bit_length {
                        return count;
                    }
                    xy_cmp = compare_no_leading_zeros(&x[x_start..], &y[y_start..]);
                    if xy_cmp <= 0 {
                        break;
                    }
                }
            }

            // NB: The case where c[cStart] is 1-bit is harmless
            shift = c_bit_length as isize - x_bit_length as isize;
            if shift == 1 {
                let first_c = c[c_start] >> 1;
                let first_x = x[x_start as usize];
                if first_c > first_x {
                    shift += 1;
                }
            }
            if shift < 2 {
                shift_right_one_in_place(&mut c[c_start..]);
                c_bit_length -= 1;
                shift_right_one_in_place(&mut icount[i_count_start..])
            } else {
                shift_right_in_place(&mut c[c_start..], shift);
                c_bit_length -= shift as usize;
                shift_right_in_place(&mut icount[i_count_start..], shift);
            }

            while c[c_start] == 0 {
                c_start += 1;
            }

            while icount[i_count_start] == 0 {
                i_count_start += 1;
            }
        }
    } else {
        count = vec![0u32];
    }
    if xy_cmp == 0 {
        add_magnitudes(count.as_mut_slice(), &(*ONE).magnitude);
        for i in &mut x[x_start as usize..] {
            *i = 0;
        }
    }
    count
}

/// do a left shift - this returns a new array.
fn shift_left(mag: &[u32], n: usize) -> Vec<u32> {
    let n_ints = n >> 5;
    let n_bits = n & 0x1F;
    let mag_len = mag.len();
    let mut new_mag: Vec<u32>;

    if n_bits == 0 {
        new_mag = vec![0u32; mag_len + n_ints];
        new_mag[0..mag_len].copy_from_slice(mag);
    } else {
        let mut i = 0;
        let n_bits2 = 32 - n_bits;
        let high_bits = mag[0] >> n_bits2;

        if high_bits != 0 {
            new_mag = vec![0u32; mag_len + n_ints + 1];
            new_mag[i] = high_bits;
            i += 1;
        } else {
            new_mag = vec![0u32; mag_len + n_ints];
        }

        let mut m = mag[0];
        for j in 0..(mag_len - 1) {
            let next = mag[j + 1];
            new_mag[i] = (m << n_bits) | (next >> n_bits2);
            i += 1;
            m = next;
        }
        new_mag[i] = mag[mag_len - 1] << n_bits;
    }
    new_mag
}

// do a right shift by one - this does it in place.
fn shift_right_one_in_place(mag: &mut [u32]) {
    let mut i = mag.len();
    let mut m = mag[i - 1];

    while ({
        i -= 1;
        i
    } > 0)
    {
        let next = mag[i - 1];
        mag[i] = (m >> 1) | (next << 31);
        m = next;
    }

    mag[0] = mag[0] >> 1;
}

// do a right shift - this does it in place.
fn shift_right_in_place(mag: &mut [u32], n: isize) {
    let n_ints = (n as usize >> 5) as isize;
    let n_bits = n & 0x1F;
    let mag_end = mag.len() - 1;

    if n_ints != 0 {
        let delta = n_ints;
        for i in (n_ints as usize..=mag_end).rev() {
            mag[i] = mag[i - delta as usize];
        }
        for i in (0..n_ints).rev() {
            mag[i as usize] = 0;
        }
    }

    if n_bits != 0 {
        let n_bits2 = 32 - n_bits;
        let mut m = mag[mag_end];

        for i in ((n_ints + 1) as usize..=mag_end).rev() {
            let next = mag[i - 1];
            mag[i] = (m >> n_bits) | (next << n_bits2);
            m = next;
        }
        mag[n_ints as usize] = mag[n_ints as usize] >> n_bits;
    }
}

/// return x = x % y - done in place (y value preserved)
fn remainder(x: &mut [u32], y: &[u32]) {
    let mut x_start = 0;
    while x_start < x.len() && x[x_start] == 0 {
        x_start += 1;
    }
    let mut y_start = 0;
    while y_start < y.len() && y[y_start] == 0 {
        y_start += 1;
    }

    debug_assert!(y_start < y.len());

    let mut xy_cmp = compare_no_leading_zeros(&x[x_start..], &y[y_start..]);
    if xy_cmp > 0 {
        let y_bit_length = calc_bit_length(1, &y[y_start..]);
        let mut x_bit_length = calc_bit_length(1, &x[x_start..]);
        let mut shift = x_bit_length as isize - y_bit_length as isize;

        let mut c: Vec<u32>;
        let mut c_start = 0;
        let mut c_bit_length = y_bit_length;
        if shift > 0 {
            c = shift_left(y, shift as usize);
            c_bit_length += shift as usize;
            debug_assert!(c[0] != 0);
        } else {
            let len = y.len() - y_start;

            c = vec![0u32; len];
            c.copy_from_slice(&y[y_start..]);
        }
        loop {
            if c_bit_length < x_bit_length
                || compare_no_leading_zeros(&x[x_start..], &c[c_start..]) >= 0
            {
                subtract(&mut x[x_start..], &c[c_start..]);
                while x[x_start] == 0 {
                    x_start += 1;
                    if x_start == x.len() {
                        return;
                    }
                }

                x_bit_length = 32 * (x.len() - x_start - 1) + bit_len(x[x_start]) as usize;
                if x_bit_length <= y_bit_length {
                    if x_bit_length < y_bit_length {
                        return;
                    }

                    xy_cmp = compare_no_leading_zeros(&x[x_start..], &y[y_start..]);
                    if xy_cmp <= 0 {
                        break;
                    }
                }
            }

            shift = c_bit_length as isize - x_bit_length as isize;
            if shift == 1 {
                let first_c = c[c_start] >> 1;
                let first_x = x[x_start];
                if first_c > first_x {
                    shift += 1;
                }
            }

            if shift < 2 {
                shift_right_one_in_place(&mut c[c_start..]);
                c_bit_length -= 1;
            } else {
                shift_right_in_place(&mut c[c_start..], shift);
                c_bit_length -= shift as usize;
            }

            while c[c_start] == 0 {
                c_start += 1;
            }
        }
    }
    if xy_cmp == 0 {
        for i in &mut x[x_start..] {
            *i = 0;
        }
    }
}

fn create_value_of_u32(value: u32) -> BigInteger {
    if value == 0 {
        return (*ZERO).clone();
    }
    BigInteger::with_check_mag(1, Arc::new(vec![value]), false)
}

fn append_zero_extended_string(sb: &mut String, s: &str, min_length: usize) {
    let mut len = s.len();
    while len < min_length {
        sb.push('0');
        len += 1;
    }
    sb.push_str(s);
}

// return w with w = x * x - w is assumed to have enough space.
fn square(w: &mut [u32], x: &[u32]) {
    // Note: this method allows w to be only (2 * x.Length - 1) words if result will fit
    // if w.len() != 2 * x.len() {
    //     panic!("no I don't think so...");
    // }
    let mut c: u64;
    let mut w_base: i32 = (w.len() - 1) as i32;
    for i in (1..x.len()).rev() {
        let v = x[i] as u64;
        c = v * v + w[w_base as usize] as u64;
        w[w_base as usize] = c as u32;
        c >>= 32;

        for j in (0..i).rev() {
            let prod = v * x[j] as u64;
            c += (w[{
                w_base -= 1;
                w_base as usize
            }] as u64
                & UIMASK)
                + (((prod as u32) << 1) as u64);
            w[w_base as usize] = c as u32;
            c = (c >> 32) + (prod >> 31);
        }

        c += w[{
            w_base -= 1;
            w_base as usize
        }] as u64;
        w[w_base as usize] = c as u32;

        if {
            w_base -= 1;
            w_base
        } >= 0
        {
            w[w_base as usize] = (c >> 32) as u32;
        } else {
            debug_assert!(c >> 32 == 0);
        }

        w_base += i as i32;
    }

    c = x[0] as u64;
    c = c * c + w[w_base as usize] as u64;
    w[w_base as usize] = c as u32;

    if {
        w_base -= 1;
        w_base
    } >= 0
    {
        w[w_base as usize] += (c >> 32) as u32;
    } else {
        debug_assert!(c >> 32 == 0);
    }
}

fn to_string_with_moduli(
    sb: &mut String,
    radix: u32,
    moduli: &[BigInteger],
    mut scale: usize,
    pos: &BigInteger,
) {
    if *pos.bit_length() < 64 {
        let s = match radix {
            2 => format!("{:b}", pos.get_i64_value()),
            8 => format!("{:o}", pos.get_i64_value()),
            10 => format!("{}", pos.get_i64_value()),
            16 => format!("{:X}", pos.get_i64_value()),
            _ => panic!("Only bases 2, 8, 10, 16 are allowed"),
        };
        if sb.len() > 1 || sb.len() == 1 && &sb[0..1] != "-" {
            append_zero_extended_string(sb, &s, 1 << scale);
        } else if pos.sign != 0 {
            sb.push_str(&s);
        }
        return;
    }
    scale -= 1;
    let (qr1, qr2) = pos.divide_and_remainder(&moduli[scale]);
    to_string_with_moduli(sb, radix, moduli, scale, &qr1);
    to_string_with_moduli(sb, radix, moduli, scale, &qr2);
}

fn get_bytes_length(n_bits: usize) -> usize {
    (n_bits + BITS_PER_BYTE - 1) / BITS_PER_BYTE
}

fn get_window_list(mag: &[u32], extra_bits: usize) -> Vec<u32> {
    let mut v = mag[0];
    debug_assert!(v != 0);
    let leading_bits = bit_len(v);
    let total_bits = ((mag.len() - 1) << 5) + leading_bits as usize;
    let result_size = (total_bits + extra_bits) / (1 + extra_bits) + 1;
    let mut result = vec![0u32; result_size];
    let mut result_pos = 0;
    let mut bit_pos = 33 - leading_bits;
    v = v.wrapping_shl(bit_pos);

    let mut mult = 1;
    let mult_limit = 1 << extra_bits;
    let mut zeros = 0;

    let mut i = 0;
    loop {
        while bit_pos < 32 {
            bit_pos += 1;

            if mult < mult_limit {
                mult = (mult << 1) | (v >> 31);
            } else if (v as i32) < 0 {
                result[result_pos] = create_window_entry(mult, zeros);
                result_pos += 1;
                mult = 1;
                zeros = 0;
            } else {
                zeros += 1;
            }

            v <<= 1;
        }

        i += 1;
        if i == mag.len() {
            result[result_pos] = create_window_entry(mult, zeros);
            result_pos += 1;
            break;
        }

        v = mag[i];
        bit_pos = 0;
    }

    result[result_pos] = u32::MAX;
    result
}

fn create_window_entry(mut mult: u32, mut zeros: u32) -> u32 {
    debug_assert!(mult > 0);
    let tz = mult.trailing_zeros();
    mult >>= tz;
    zeros += tz;
    mult | (zeros << 8)
}

/// m_dash = -m ^ (-1) mod b
fn square_monty(a: &mut [u32], x: &mut [u32], m: &[u32], m_dash: u32, small_monty_modulus: bool) {
    let n = m.len() as i32;
    if n == 1 {
        let x_val = x[0];
        x[0] = multiply_monty_n_is_one(x_val, x_val, m[0], m_dash);
        return;
    }

    let x0 = x[(n - 1) as usize] as u64;
    let mut a_max: u32;
    {
        let mut carry = x0 * x0;
        let t = (carry as u32).wrapping_mul(m_dash) as u64;

        let mut prod2 = t.wrapping_mul(m[(n - 1) as usize] as u64);
        carry += (prod2 as u32) as u64;
        debug_assert!(carry as u32 == 0);
        carry = (carry >> 32) + (prod2 >> 32);

        for j in (0..=(n - 2)).rev() {
            let prod1 = x0 * x[j as usize] as u64;
            prod2 = t.wrapping_mul(m[j as usize] as u64);

            carry += (prod2 & UIMASK) + ((prod1 as u32) << 1) as u64;
            a[(j + 2) as usize] = carry as u32;
            carry = (carry >> 32) + (prod1 >> 31) + (prod2 >> 32);
        }

        a[1] = carry as u32;
        a_max = (carry >> 32) as u32;
    }

    for i in (0..=n - 2).rev() {
        let a0 = a[n as usize];
        let t = a0.wrapping_mul(m_dash) as u64;
        let mut carry = t
            .wrapping_mul(m[(n - 1) as usize] as u64)
            .wrapping_add(a0 as u64);
        debug_assert!(carry as u32 == 0);
        carry >>= 32;

        for j in ((i + 1)..=(n - 2)).rev() {
            carry += t * m[j as usize] as u64 + a[(j + 1) as usize] as u64;
            a[(j + 2) as usize] = carry as u32;
            carry >>= 32;
        }

        let xi = x[i as usize] as u64;
        {
            let prod1 = xi * xi;
            let prod2 = t.wrapping_mul(m[i as usize] as u64);

            carry += (prod1 & UIMASK) + (prod2 as u32) as u64 + a[(i + 1) as usize] as u64;
            a[(i + 2) as usize] = carry as u32;
            carry = (carry >> 32) + (prod1 >> 32) + (prod2 >> 32);
        }

        for j in (0..=(i - 1)).rev() {
            let prod1 = xi * x[j as usize] as u64;
            let prod2 = t * m[j as usize] as u64;

            carry += (prod2 & UIMASK) + ((prod1 as u32) << 1) as u64 + a[(j + 1) as usize] as u64;
            a[(j + 2) as usize] = carry as u32;
            carry = (carry >> 32) + (prod1 >> 31) + (prod2 >> 32);
        }

        carry += a_max as u64;
        a[1] = carry as u32;
        a_max = (carry >> 32) as u32;
    }

    a[0] = a_max;

    if !small_monty_modulus && compare_to(&a, &m) >= 0 {
        subtract(a, &m);
    }

    x[0..n as usize].copy_from_slice(&a[1..(1 + n) as usize])
}

fn multiply_monty_n_is_one(x: u32, y: u32, m: u32, m_dash: u32) -> u32 {
    let mut carry = x as u64 * y as u64;
    let t = (carry as u32).wrapping_mul(m_dash);
    let um = m as u64;
    let prod2 = um * t as u64;
    carry += (prod2 as u32) as u64;
    debug_assert!(carry as u32 == 0);
    carry = (carry >> 32) + (prod2 >> 32);
    if carry > um {
        carry -= um;
    }
    debug_assert!(carry < um);
    carry as u32
}

fn compare_to(x: &[u32], y: &[u32]) -> i32 {
    let mut x_index = 0usize;
    while x_index != x.len() && x[x_index] == 0 {
        x_index += 1;
    }
    let mut y_index = 0usize;
    while y_index != y.len() && y[y_index] == 0 {
        y_index += 1;
    }
    compare_no_leading_zeros(&x[x_index..], &y[y_index..])
}

/// Montgomery multiplication: a = x * y * R^(-1) mod m
/// Based algorithm 14.36 of Handbook of Applied Cryptography.
/// - m, x, y should have length n
/// - a should have length (n + 1)
/// - b = 2^32, R = b^n
/// The result is put in x
/// NOTE: the indices of x, y, m, a different in HAC and in Java
fn multiply_monty(
    a: &mut [u32],
    x: &mut [u32],
    y: &[u32],
    m: &[u32],
    m_dash: u32,
    small_monty_modulus: bool,
) {
    let n = m.len();
    if n == 1 {
        x[0] = multiply_monty_n_is_one(x[0], y[0], m[0], m_dash);
        return;
    }

    let y0 = y[n - 1];
    let mut a_max: u32;
    {
        let xi = x[n - 1] as u64;

        let mut carry = xi * y0 as u64;
        let t = (carry as u32).wrapping_mul(m_dash) as u64;

        let mut prod2 = t.wrapping_mul(m[n - 1] as u64);
        carry += (prod2 as u32) as u64;
        debug_assert!(carry as u32 == 0);
        carry = (carry >> 32) + (prod2 >> 32);

        for j in (0..=(n - 2)).rev() {
            let prod1 = xi as u64 * y[j] as u64;
            prod2 = t.wrapping_mul(m[j] as u64);

            carry += (prod1 & UIMASK) + (prod2 as u32) as u64;
            a[j + 2] = carry as u32;
            carry = (carry >> 32) + (prod1 >> 32) + (prod2 >> 32);
        }

        a[1] = carry as u32;
        a_max = (carry >> 32) as u32;
    }

    for i in (0..=(n - 2)).rev() {
        let a0 = a[n];
        let xi = x[i] as u64;
        let mut prod1 = xi * y0 as u64;
        let mut carry = (prod1 & UIMASK) + a0 as u64;
        let t = (carry as u32).wrapping_mul(m_dash) as u64;

        let mut prod2 = t.wrapping_mul(m[n - 1] as u64);
        carry += (prod2 as u32) as u64;
        debug_assert!(carry as u32 == 0);
        carry = (carry >> 32) + (prod1 >> 32) + (prod2 >> 32);

        for j in (0..=(n - 2)).rev() {
            prod1 = xi as u64 * y[j] as u64;
            prod2 = t.wrapping_mul(m[j] as u64);

            carry += (prod1 & UIMASK) + (prod2 as u32) as u64 + a[j + 1] as u64;
            a[j + 2] = carry as u32;
            carry = (carry >> 32) + (prod1 >> 32) + (prod2 >> 32);
        }

        carry += a_max as u64;
        a[1] = carry as u32;
        a_max = (carry >> 32) as u32;
    }

    a[0] = a_max;
    if !small_monty_modulus && compare_to(&a, &m) >= 0 {
        subtract(a, m);
    }
    x[0..n].copy_from_slice(&a[1..(n + 1)]);
}

/// mDash = -m^(-1) mod b
fn montgomery_reduce(x: &mut [u32], m: &[u32], m_dash: u32) {
    // NOTE: Not a general purpose reduction (which would allow x up to twice the bitlength of m)
    debug_assert!(x.len() == m.len());

    let n = m.len();
    for _ in (0..=(n - 1)).rev() {
        let x0 = x[n - 1];
        let t = x0.wrapping_mul(m_dash) as u64;

        let mut carry = t * m[n - 1] as u64 + x0 as u64;
        debug_assert!(carry as u32 == 0);

        carry >>= 32;

        if n as i32 - 2 >= 0 {
            for j in (0..=(n - 2)).rev() {
                carry += t * m[j] as u64 + x[j] as u64;
                x[j + 1] = carry as u32;
                carry >>= 32;
            }
        }

        x[0] = carry as u32;
        debug_assert!(carry >> 32 == 0);
    }

    if compare_to(x, m) > 0 {
        subtract(x, m);
    }
}

/// return x with x = y * z - x is assumed to have enough space.
fn multiply(x: &mut [u32], y: &[u32], z: &[u32]) {
    let mut i = z.len();
    if i < 1 {
        return;
    }

    let mut x_base = x.len() as isize - y.len() as isize;

    loop {
        i -= 1;
        let a = z[i] as i64 & IMASK;
        let mut val = 0i64;

        if a != 0 {
            for j in (0..y.len()).rev() {
                val += a.wrapping_mul(y[j] as i64 & IMASK)
                    + (x[(x_base + j as isize) as usize] as i64 & IMASK);
                x[(x_base + j as isize) as usize] = val as u32;
                val = ((val as u64) >> 32) as i64;
            }
        }

        x_base -= 1;

        if x_base >= 0 {
            x[x_base as usize] = val as u32;
        } else {
            debug_assert!(val == 0);
        }
        if i > 0 {
            // nothing
        } else {
            break;
        }
    }
}

/// The first few odd primes  
/// Each list has a product < 2^31
pub(crate) static PRIME_LISTS: LazyLock<Vec<Vec<u32>>> = LazyLock::new(|| {
    return vec![
        vec![3, 5, 7, 11, 13, 17, 19, 23],
        vec![29, 31, 37, 41, 43],
        vec![47, 53, 59, 61, 67],
        vec![71, 73, 79, 83],
        vec![89, 97, 101, 103],
        vec![107, 109, 113, 127],
        vec![131, 137, 139, 149],
        vec![151, 157, 163, 167],
        vec![173, 179, 181, 191],
        vec![193, 197, 199, 211],
        vec![223, 227, 229],
        vec![233, 239, 241],
        vec![251, 257, 263],
        vec![269, 271, 277],
        vec![281, 283, 293],
        vec![307, 311, 313],
        vec![317, 331, 337],
        vec![347, 349, 353],
        vec![359, 367, 373],
        vec![379, 383, 389],
        vec![397, 401, 409],
        vec![419, 421, 431],
        vec![433, 439, 443],
        vec![449, 457, 461],
        vec![463, 467, 479],
        vec![487, 491, 499],
        vec![503, 509, 521],
        vec![523, 541, 547],
        vec![557, 563, 569],
        vec![571, 577, 587],
        vec![593, 599, 601],
        vec![607, 613, 617],
        vec![619, 631, 641],
        vec![643, 647, 653],
        vec![659, 661, 673],
        vec![677, 683, 691],
        vec![701, 709, 719],
        vec![727, 733, 739],
        vec![743, 751, 757],
        vec![761, 769, 773],
        vec![787, 797, 809],
        vec![811, 821, 823],
        vec![827, 829, 839],
        vec![853, 857, 859],
        vec![863, 877, 881],
        vec![883, 887, 907],
        vec![911, 919, 929],
        vec![937, 941, 947],
        vec![953, 967, 971],
        vec![977, 983, 991],
        vec![997, 1009, 1013],
        vec![1019, 1021, 1031],
        vec![1033, 1039, 1049],
        vec![1051, 1061, 1063],
        vec![1069, 1087, 1091],
        vec![1093, 1097, 1103],
        vec![1109, 1117, 1123],
        vec![1129, 1151, 1153],
        vec![1163, 1171, 1181],
        vec![1187, 1193, 1201],
        vec![1213, 1217, 1223],
        vec![1229, 1231, 1237],
        vec![1249, 1259, 1277],
        vec![1279, 1283, 1289],
    ];
});

pub(crate) static PRIME_PRODUCTS: LazyLock<Vec<u32>> = LazyLock::new(|| {
    let lists = &(*PRIME_LISTS);
    let mut result = vec![0u32; lists.len()];
    for i in 0..lists.len() {
        let prime_list = &lists[i];
        let mut product = prime_list[0];
        for j in 1..prime_list.len() {
            product *= prime_list[j];
        }
        result[i] = product;
    }
    result
});

/// These are the threshold bit-lengths (of an exponent) where we increase the window size.
/// They are calculated according to the expected savings in multiplications.
/// Some squares will also be saved on average, but we offset these against the extra storage costs.
static EXP_WINDOW_THRESHOLDS: [usize; 8] = [7, 25, 81, 241, 673, 1793, 4609, i32::MAX as usize];

/// Calculate the numbers u1, u2, and u3 such that:  
///
/// u1 * a + u2 * b = u3  
///
/// where u3 is the greatest common divider of a and b. a and b using the extended Euclid algorithm
/// (refer p. 323 of The Art of Computer Programming vol 2, 2nd ed).
/// This also seems to have the side effect of calculating some form of multiplicative inverse.  
fn ext_euclid(a: &BigInteger, b: &BigInteger) -> (BigInteger, BigInteger) {
    let mut u1 = (*ONE).clone();
    let mut v1 = (*ZERO).clone();
    let mut u3 = a.clone();
    let mut v3 = b.clone();

    if v3.get_sign_value() > 0 {
        loop {
            let q = u3.divide_and_remainder(&v3);
            u3 = v3;
            v3 = q.1;

            let old_u1 = u1;
            u1 = v1.clone();
            if v3.get_sign_value() <= 0 {
                break;
            }
            v1 = old_u1.subtract(&v1.multiply(&q.0));
        }
    }
    (u3, u1)
}

#[cfg(test)]
mod test {
    use super::make_magnitude_le;
    use super::strip_prefix_value;
    use super::C_ZERO_MAGNITUDE;

    #[test]
    fn test_01_strip_prefix_value() {
        let bs = [0x00, 0x00, 0x00, 0x00];
        assert_eq!(strip_prefix_value(&bs, 0x0).len(), 0);
    }
    #[test]
    fn test_02_strip_prefix_value() {
        let bs = [0x00, 0x00, 0x01, 0x00];
        assert_eq!(strip_prefix_value(&bs, 0x0), [0x01, 0x00]);
    }
    #[test]
    fn test_03_strip_prefix_value() {
        let bs = [0x01, 0x00, 0x01, 0x04];
        assert_eq!(strip_prefix_value(&bs, 0x0), &bs);
    }

    #[test]
    fn test_01_make_magnitude_le() {
        let bs = [0x00, 0x00, 0x00, 0x00];
        assert_eq!(make_magnitude_le(&bs), C_ZERO_MAGNITUDE);
    }
    #[test]
    fn test_02_make_magnitude_le() {
        let bs = [0x01, 0x02, 0x03, 0x04, 0x05, 0x6, 0x7, 0x8, 0x9, 0x00];
        assert_eq!(make_magnitude_le(&bs), vec![0x09, 0x08070605, 0x04030201]);
    }
}
