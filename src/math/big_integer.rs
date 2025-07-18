use crate::math::raw::internal_mod::{inverse_u32, inverse_u64};
use crate::util::pack::{be_to_u32_buffer, be_to_u32_low, le_to_u32_low, u32_to_be_bytes};
use crate::{BcError, Result};
use rand::prelude::*;
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::ops::Neg;
use std::sync::{LazyLock, OnceLock};

const I_MASK: i64 = 0xFFFFFFFF;
const U_I_MASK: u64 = 0xFFFFFFFF;
const C_BYTES_PRE_INT: usize = size_of::<u32>();
const BITS_PER_INT: usize = C_BYTES_PRE_INT * 8;
const BITS_PER_BYTE: usize = size_of::<u8>() * 8;
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
#[derive(Clone, Debug, Eq)]
pub struct BigInteger {
    sign: i8,
    magnitude: Vec<u32>,
    bits: OnceLock<usize>,
    bit_length: OnceLock<usize>,
}

impl BigInteger {
    fn new(sign: i8, magnitude: Vec<u32>) -> Self {
        BigInteger {
            sign,
            magnitude,
            bits: OnceLock::new(),
            bit_length: OnceLock::new(),
        }
    }
    /// Creates a new `BigInteger` from a `u8` value.
    /// # Arguments
    /// * `value`: A `u8` value to create the `BigInteger` from.
    /// # Returns
    /// A `BigInteger` instance representing the given `u8` value.
    /// # Example
    /// ```rust
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_u8(42);
    /// assert_eq!(big_int.as_u8(), 42);
    /// ```
    pub fn with_u8(value: u8) -> Self {
        Self::with_u32(value as u32)
    }
    /// Creates a new `BigInteger` from a `u16` value.
    /// # Arguments
    /// * `value`: A `u16` value to create the `BigInteger` from.
    /// # Returns
    /// A `BigInteger` instance representing the given `u16` value.
    /// # Example
    /// ```rust
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_u16(42);
    /// assert_eq!(big_int.as_u16(), 42);
    /// ```
    pub fn with_u16(value: u16) -> Self {
        Self::with_u32(value as u32)
    }
    /// Creates a new `BigInteger` from a `u32` value.
    /// # Arguments
    /// * `value`: A `u32` value to create the `BigInteger` from.
    /// # Returns
    /// A `BigInteger` instance representing the given `u32` value.
    /// # Example
    /// ```rust
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_u32(42);
    /// assert_eq!(big_int.as_u32(), 42);
    /// ```
    pub fn with_u32(value: u32) -> Self {
        if value == 0 {
            return BigInteger::new(0, vec![]);
        }
        Self::new(1, vec![value])
    }
    /// Creates a new `BigInteger` from a `u64` value.
    /// # Arguments
    /// * `value` - A `u64` value to create the `BigInteger` from.
    /// # Returns
    /// A `BigInteger` instance representing the given `u64` value.
    /// # Example
    /// ```
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_u64(u64::MAX - 1);
    /// assert_eq!(big_int.as_u64(), u64::MAX - 1);
    /// ```
    pub fn with_u64(value: u64) -> Self {
        if value == 0 {
            return BigInteger::new(0, vec![]);
        }
        let msw = (value >> 32) as u32;
        let lsw = value as u32;
        if msw == 0 {
            Self::new(1, vec![lsw])
        } else {
            Self::new(1, vec![msw, lsw])
        }
    }
    /// Creates a new `BigInteger` from a `u128` value.
    /// # Arguments
    /// * `value` - A `u128` value to create the `BigInteger` from.
    /// # Returns
    /// A `BigInteger` instance representing the given `u128` value.
    /// # Example
    /// ```
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_u128(u128::MAX - 1);
    /// assert_eq!(big_int.as_u128(), u128::MAX - 1);
    /// ```
    pub fn with_u128(value: u128) -> Self {
        if value == 0 {
            return BigInteger::new(0, vec![]);
        }
        let b0 = ((value >> 96) & 0xFFFFFFFF) as u32;
        let b1 = ((value >> 64) & 0xFFFFFFFF) as u32;
        let b2 = ((value >> 32) & 0xFFFFFFFF) as u32;
        let b3 = ((value >> 0) & 0xFFFFFFFF) as u32;

        if b0 == 0 && b1 == 0 && b2 == 0 {
            Self::new(1, vec![b3])
        } else if b0 == 0 && b1 == 0 {
            Self::new(1, vec![b2, b3])
        } else if b0 == 0 {
            Self::new(1, vec![b1, b2, b3])
        } else {
            Self::new(1, vec![b0, b1, b2, b3])
        }
    }
    /// Creates a new `BigInteger` from an `i8` value.
    ///
    /// # Arguments
    /// * `value` - An `i8` value to create the `BigInteger` from.
    ///
    /// # Returns
    /// A `BigInteger` instance representing the given `i8` value.
    ///
    /// # Example
    /// ```
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_i8(-42);
    /// assert_eq!(big_int.as_i8(), -42);
    /// ```
    pub fn with_i8(value: i8) -> Self {
        if value == 0 {
            return BigInteger::new(0, vec![]);
        }
        if value < 0 {
            BigInteger::new(-1, vec![value.unsigned_abs() as u32])
        } else {
            Self::with_u32(value as u32)
        }
    }
    /// Creates a new `BigInteger` from an `i16` value.
    ///
    /// # Arguments
    /// * `value` - An `i16` value to create the `BigInteger` from.
    ///
    /// # Returns
    /// A `BigInteger` instance representing the given `i16` value.
    ///
    /// # Example
    /// ```
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_i16(-42);
    /// assert_eq!(big_int.as_i16(), -42);
    /// ```
    pub fn with_i16(value: i16) -> Self {
        if value == 0 {
            return BigInteger::new(0, vec![]);
        }
        if value < 0 {
            BigInteger::new(-1, vec![value.unsigned_abs() as u32])
        } else {
            Self::with_u32(value as u32)
        }
    }
    /// Creates a new `BigInteger` from an `i32` value.
    ///
    /// # Arguments
    /// * `value` - An `i32` value to create the `BigInteger` from.
    ///
    /// # Returns
    /// A `BigInteger` instance representing the given `i32` value.
    ///
    /// # Example
    /// ```
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_i32(-42);
    /// assert_eq!(big_int.as_i32(), -42);
    /// ```
    pub fn with_i32(value: i32) -> Self {
        if value >= 0 {
            if (value as usize) < SMALL_CONSTANTS.len() {
                return SMALL_CONSTANTS[value as usize].clone();
            }
            return create_value_of_u32(value as u32);
        }
        if value == i32::MIN {
            return create_value_of_u32(!value as u32).not();
        }
        Self::with_i32(-value).negate()
    }
    /// Creates a new `BigInteger` from an `i64` value.
    ///
    /// # Arguments
    /// * `value` - An `i64` value to create the `BigInteger` from.
    ///
    /// # Returns
    /// A `BigInteger` instance representing the given `i64` value.
    ///
    /// # Example
    /// ```
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_i64(-42);
    /// assert_eq!(big_int.as_i64(), -42);
    /// ```
    pub fn with_i64(value: i64) -> Self {
        if value >= 0 {
            if value < SMALL_CONSTANTS.len() as i64 {
                return SMALL_CONSTANTS[value as usize].clone();
            }
            return Self::with_u64(value as u64);
        }

        if value == i64::MIN {
            return Self::with_u64(!value as u64).not();
        }
        Self::with_i64(-value).negate()
    }
    /// Creates a new `BigInteger` from an `i128` value.
    ///
    /// # Arguments
    /// * `value` - An `i128` value to create the `BigInteger` from.
    ///
    /// # Returns
    /// A `BigInteger` instance representing the given `i128` value.
    ///
    /// # Example
    /// ```
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_i128(-566000);
    /// assert_eq!(big_int.as_i128(), -566000);
    /// ```
    pub fn with_i128(value: i128) -> Self {
        if value == 0 {
            return BigInteger::new(0, vec![]);
        }
        if value < 0 {
            let abs_value = value.unsigned_abs();
            let b0 = ((abs_value >> 96) & 0xFFFFFFFF) as u32;
            let b1 = ((abs_value >> 64) & 0xFFFFFFFF) as u32;
            let b2 = ((abs_value >> 32) & 0xFFFFFFFF) as u32;
            let b3 = ((abs_value >> 0) & 0xFFFFFFFF) as u32;

            if b0 == 0 && b1 == 0 && b2 == 0 {
                BigInteger::new(-1, vec![b3])
            } else if b0 == 0 && b1 == 0 {
                BigInteger::new(-1, vec![b2, b3])
            } else if b0 == 0 {
                BigInteger::new(-1, vec![b1, b2, b3])
            } else {
                BigInteger::new(-1, vec![b0, b1, b2, b3])
            }
        } else {
            Self::with_u128(value as u128)
        }
    }
    /// Creates a new `BigInteger` from a byte buffer, interpreting it as big-endian.
    ///
    /// # Arguments
    /// * `buffer` - A byte slice representing the integer value in big-endian order.
    ///
    /// # Returns
    /// A `BigInteger` instance representing the value in the buffer.
    ///
    /// # Example
    /// ```rust
    /// use bc_rust::math::BigInteger;
    ///
    /// let buffer = [0x00, 0x00, 0x00, 0x2A]; // Represents integer 42 in big-endian
    /// let big_int = BigInteger::with_buffer(&buffer);
    /// assert_eq!(big_int.as_u32(), 42); // Should be 42
    ///
    /// let empty_buffer: [u8; 0] = [];
    /// let big_int_empty = BigInteger::with_buffer(&empty_buffer);
    /// assert_eq!(big_int_empty.as_u32(), 0); // Should be zero
    ///
    /// let zero_buffer = [0x00, 0x00, 0x00, 0x00]; // Represents zero
    /// let big_int_zero = BigInteger::with_buffer(&zero_buffer);
    /// assert_eq!(big_int_zero.as_u32(), 0); // Should be zero
    ///
    /// let negative_buffer = [0xFF, 0xFF, 0xFF, 0xD6]; // Represents -42 in big-endian
    /// let big_int_negative = BigInteger::with_buffer(&negative_buffer);
    /// assert_eq!(big_int_negative.as_i32(), -42); // Should be -42
    /// ```
    pub fn with_buffer(buffer: &[u8]) -> Self {
        Self::with_buffer_big_endian(buffer, true)
    }
    /// Creates a new `BigInteger` from a byte buffer, specifying endianness.
    ///
    /// # Arguments
    /// * `buffer` - A byte slice representing the integer value.
    /// * `big_endian` - If `true`, interprets the buffer as big-endian; otherwise, little-endian.
    ///
    /// # Returns
    /// A `BigInteger` instance representing the value in the buffer.
    ///
    /// # Example
    /// ```rust
    /// use bc_rust::math::BigInteger;
    ///
    /// let buffer = [0x00, 0x00, 0x00, 0x2A]; // Represents integer 42 in big-endian
    /// let big_int = BigInteger::with_buffer_big_endian(&buffer, true);
    /// assert_eq!(big_int.as_u32(), 42); // Should be 42
    ///
    /// let buffer_le = [0x2A, 0x00, 0x00, 0x00]; // Represents integer 42 in little-endian
    /// let big_int_le = BigInteger::with_buffer_big_endian(&buffer_le, false);
    /// assert_eq!(big_int_le.as_u32(), 42); // Should also be 42
    ///
    /// let empty_buffer: [u8; 0] = [];
    /// let big_int_empty = BigInteger::with_buffer_big_endian(&empty_buffer, true);
    /// assert_eq!(big_int_empty.as_u32(), 0); // Should be zero
    ///
    /// let zero_buffer = [0x00, 0x00, 0x00, 0x00]; // Represents zero
    /// let big_int_zero = BigInteger::with_buffer_big_endian(&zero_buffer, true);
    /// assert_eq!(big_int_zero.as_u32(), 0); // Should be zero
    ///
    /// let negative_buffer = [0xFF, 0xFF, 0xFF, 0xD6]; // Represents -42 in big-endian
    /// let big_int_negative = BigInteger::with_buffer_big_endian(&negative_buffer, true);
    /// assert_eq!(big_int_negative.as_i32(), -42); // Should be -42
    ///
    /// let negative_buffer_le = [0xD6, 0xFF, 0xFF, 0xFF]; // Represents -42 in little-endian
    /// let big_int_negative_le = BigInteger::with_buffer_big_endian(&negative_buffer_le, false);
    /// assert_eq!(big_int_negative_le.as_i32(), -42); // Should also be -42
    /// ```
    pub fn with_buffer_big_endian(buffer: &[u8], big_endian: bool) -> Self {
        if buffer.is_empty() {
            return BigInteger::new(0, vec![]);
        }
        let result = if big_endian { init_be(buffer) } else { init_le(buffer) };
        Self::new(result.0, result.1)
    }
    pub fn with_sign_buffer(sign: i8, buffer: &[u8]) -> Result<Self> {
        Self::with_sign_buffer_big_endian(sign, buffer, true)
    }
    /// # Arguments
    /// * `sign` - The sign of the integer.
    /// * `buffer` - The buffer containing the integer.
    /// * `big_endian` - The endianness of the buffer.
    ///
    /// # Returns
    /// A `BigInteger` instance.
    ///
    /// # Errors
    /// Returns `Error` if the sign is invalid.
    pub fn with_sign_buffer_big_endian(sign: i8, buffer: &[u8], big_endian: bool) -> Result<Self> {
        if sign < -1 || sign > 1 {
            return Err(BcError::with_invalid_argument("Sign must be -1, 0, or 1"));
        }

        if sign == 0 {
            return Ok((*ZERO).clone());
        }
        let magnitude = if big_endian {
            make_magnitude_be(buffer)
        } else {
            make_magnitude_le(buffer)
        };
        Ok(Self::new(sign, magnitude))
    }
    /// Creates a `BigInteger` from a string with the specified radix.
    ///
    /// # Parameters
    ///
    /// * `str` - A string representing the integer.
    /// * `radix` - The radix/base. It Must be 2, 8, 10, or 16.
    ///
    /// # Errors
    ///
    /// - Returns `Error` if the string is empty.
    /// - Returns `Error` if the string contains invalid characters for the specified radix.
    ///
    /// # Returns
    /// A `BigInteger` instance representing the value of the string in the specified radix.
    ///
    /// # Examples
    ///
    /// ```
    /// use bc_rust::math::BigInteger;
    /// use std::io::Error;
    ///
    /// let result_radix2 = BigInteger::with_string_radix("1010", 2).unwrap();
    /// let result_radix8 = BigInteger::with_string_radix("12", 8).unwrap();
    /// let result_radix10 = BigInteger::with_string_radix("10", 10).unwrap();
    /// let result_radix16 = BigInteger::with_string_radix("A", 16).unwrap();
    ///
    /// assert_eq!(result_radix2, result_radix8);
    /// assert_eq!(result_radix8, result_radix10);
    /// assert_eq!(result_radix10, result_radix16);
    /// ```
    pub fn with_string_radix(str: &str, radix: u32) -> Result<Self> {
        if str.is_empty() {
            return Err(BcError::with_invalid_argument("String cannot be empty"));
        }

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
                return Err(BcError::with_invalid_argument("Radix must be 2, 8, 10, or 16"));
            }
        }

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
                        if i > 1 {
                            return Err(BcError::with_invalid_argument(format!("Bad character in radix 2 string: {}", s)));
                        }
                        b = b.shift_left(1);
                    }
                    8 => {
                        if i > 8 {
                            return Err(BcError::with_invalid_argument(format!("Bad character in radix 8 string: {}", s)));
                        }
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
                    // Nothing
                } else if radix == 8 {
                    // Nothing
                } else if radix == 16 {
                    b = b.shift_left((s.chars().count() as isize) << 2);
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
    /// Creates a `BigInteger` from a string.
    ///
    /// # Parameters
    ///
    /// * `str` - A string representing the integer.
    ///
    /// # Errors
    ///
    /// - Returns `Error` if the string is empty.
    /// - Returns `Error` if the string contains invalid characters for the default radix (10).
    ///
    /// # Returns
    /// A `BigInteger` instance representing the value of the string in base 10.
    ///
    /// # Example
    /// ```rust
    /// use bc_rust::math::BigInteger;
    /// use std::io::Error;
    ///
    /// let result = BigInteger::with_string("12345").unwrap();
    /// assert_eq!(result.as_u32(), 12345);
    ///
    /// let result_neg = BigInteger::with_string("-12345").unwrap();
    /// assert_eq!(result_neg.as_i32(), -12345);
    ///
    /// let result_zero = BigInteger::with_string("0").unwrap();
    /// assert_eq!(result_zero.as_u32(), 0);
    ///
    /// let result_empty = BigInteger::with_string("");
    /// assert!(result_empty.is_err()); // Should return an error for empty string
    /// ```
    pub fn with_string(str: &str) -> Result<Self> {
        Self::with_string_radix(str, 10)
    }
    pub fn with_random(size_in_bits: usize, random: &mut dyn RngCore) -> BigInteger {
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
        BigInteger::new(sign, magnitude)
    }
    pub fn with_random_certainty(size_in_bits: usize, certainty: i32, random: &mut dyn RngCore) -> Result<Self> {
        if size_in_bits < 2 {
            return Err(BcError::with_invalid_argument("Size in bits must be at least 2"));
        }

        if size_in_bits == 2 {
            return if random.random_range(0..2) == 0 {
                Ok(Self::with_u32(2))
            } else {
                Ok(Self::with_u32(3))
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

            // ensure the trailing bit is 1 (must be odd)
            b[n_bytes - 1] |= 1;
            let mut magnitude = make_magnitude_be(&mut b);
            if certainty < 1 {
                return Ok(BigInteger::new(1, magnitude));
            }

            let result = BigInteger::new(1, magnitude.clone());
            if result.check_probable_prime(certainty, random, true)? {
                return Ok(result);
            }
            for j in 1..(magnitude.len() - 1) {
                magnitude[j] ^= random.next_u32();

                let result = BigInteger::new(1, magnitude.clone());
                if result.check_probable_prime(certainty, random, true)? {
                    return Ok(result);
                }
            }
        }
    }
    pub fn with_probable_prime(bit_length: usize, random: &mut dyn RngCore) -> Result<Self> {
        Self::with_random_certainty(bit_length, 100, random)
    }
    /// Creates a new `BigInteger` with the given sign and magnitude, optionally stripping leading zeros.
    ///
    /// # Arguments
    /// * `sign` - The sign of the number (`1` for positive, `-1` for negative, `0` for zero).
    /// * `magnitude` - The magnitude as a vector of `u32` values.
    /// * `check` - If `true`, leading zeros in the magnitude will be stripped.
    ///
    /// # Returns
    /// A `BigInteger` instance with the specified sign and magnitude.
    fn with_check_magnitude(sign: i8, magnitude: Vec<u32>, check: bool) -> Self {
        if check {
            let sub_slice = strip_prefix_value(&magnitude, 0);
            if sub_slice.is_empty() {
                Self::new(0, vec![])
            } else {
                Self::new(sign, sub_slice.to_vec())
            }
        } else {
            Self::new(sign, magnitude)
        }
    }
    pub fn sign(&self) -> i8 {
        self.sign
    }
    /// Returns the value of the `BigInteger` as a `u8`.
    pub fn as_u8(&self) -> u8 {
        if self.sign == 0 || self.magnitude.is_empty() {
            return 0;
        }
        self.magnitude[0] as u8
    }
    /// Returns the value of the `BigInteger` as a `u16`.
    pub fn as_u16(&self) -> u16 {
        if self.sign == 0 || self.magnitude.is_empty() {
            return 0;
        }
        self.magnitude[0] as u16
    }
    /// Returns the value of the `BigInteger` as a `u32`.
    pub fn as_u32(&self) -> u32 {
        if self.sign == 0 || self.magnitude.is_empty() {
            return 0;
        }
        self.magnitude[0]
    }
    /// Returns the value of the `BigInteger` as a `u64`.
    pub fn as_u64(&self) -> u64 {
        if self.sign == 0 || self.magnitude.is_empty() {
            return 0;
        }
        let mut result = 0u64;
        for &value in &self.magnitude {
            result = (result << 32) | value as u64;
        }
        result
    }
    /// Returns the value of the `BigInteger` as a `u128`.
    pub fn as_u128(&self) -> u128 {
        if self.sign == 0 || self.magnitude.is_empty() {
            return 0;
        }
        let mut result = 0u128;
        for &value in &self.magnitude {
            result = (result << 32) | value as u128;
        }
        result
    }
    /// Returns the value of the `BigInteger` as an `i8`.
    pub fn as_i8(&self) -> i8 {
        if self.sign == 0 || self.magnitude.is_empty() {
            return 0;
        }
        if self.sign < 0 {
            (self.magnitude[0] as i8).neg()
        } else {
            self.magnitude[0] as i8
        }
    }
    /// Returns the value of the `BigInteger` as an `i16`.
    pub fn as_i16(&self) -> i16 {
        if self.sign == 0 || self.magnitude.is_empty() {
            return 0;
        }
        if self.sign < 0 {
            (self.magnitude[0] as i16).neg()
        } else {
            self.magnitude[0] as i16
        }
    }
    /// Returns the value of the `BigInteger` as an `i32`.
    pub fn as_i32(&self) -> i32 {
        if self.sign == 0 {
            return 0;
        }
        let n = self.magnitude.len();
        let v = self.magnitude[n - 1] as i32;
        if self.sign < 0 { v.wrapping_neg() } else { v }
    }
    /// Returns the value of the `BigInteger` as an `i64`.
    pub fn as_i64(&self) -> i64 {
        if self.sign == 0 {
            return 0;
        }
        let n = self.magnitude.len();
        let mut v = (self.magnitude[n - 1] as i64) & I_MASK;
        if n > 1 {
            v |= ((self.magnitude[n - 2] as i64) & I_MASK) << 32;
        }
        if self.sign < 0 { v.wrapping_neg() } else { v }
    }
    /// Returns the value of the `BigInteger` as an `i128`.
    pub fn as_i128(&self) -> i128 {
        if self.sign == 0 || self.magnitude.is_empty() {
            return 0;
        }
        let mut result = 0i128;
        for &value in &self.magnitude {
            result = (result << 32) | value as i128;
        }
        if self.sign < 0 { -result } else { result }
    }
    pub fn to_string(&self) -> String {
        self.to_string_radix(10).unwrap()
    }
    pub fn to_string_radix(&self, radix: u32) -> Result<String> {
        if !(radix == 2 || radix == 8 || radix == 10 || radix == 16) {
            return Err(BcError::with_invalid_argument("Radix must be 2, 8, 10, or 16"));
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
                let mut bits = u.bit_length();
                let mut s: Vec<String> = Vec::new();
                while bits > 30 {
                    s.push(format!("{:o}", u.as_i32() & mask));
                    u = u.shift_right(30);
                    bits -= 30;
                }
                sb.push_str(&format!("{:o}", u.as_i32()));
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
            10 => {
                let q = self.abs();
                if q.bit_length() < 64 {
                    sb.push_str(&format!("{}", q.as_i64()));
                } else {
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
    pub fn to_vec(&self) -> Vec<u8> {
        self.to_vec_with_signed(false)
    }
    pub fn to_vec_unsigned(&self) -> Vec<u8> {
        self.to_vec_with_signed(true)
    }
    pub fn copy_to_u32_vec_signed_little_endian(&self, output: &mut [u32]) -> Result<()> {
        self.copy_to_u32_vec_little_endian(false, output)
    }
    pub fn copy_to_u32_vec_unsigned_little_endian(&self, output: &mut [u32]) -> Result<()> {
        self.copy_to_u32_vec_little_endian(true, output)
    }
    pub fn copy_to_u32_vec_signed_big_endian(&self, output: &mut [u32]) -> Result<()> {
        self.copy_to_u32_vec_big_endian(false, output)
    }
    pub fn copy_to_u32_vec_unsigned_big_endian(&self, output: &mut [u32]) -> Result<()> {
        self.copy_to_u32_vec_big_endian(true, output)
    }
    // operators

    /// Returns the bit length of the `BigInteger`.
    ///
    /// This method calculates and caches the number of bits required to represent the value.
    ///
    /// # Returns
    ///
    /// * `usize` - The number of bits required to represent the current value.
    /// # Example
    /// ```rust
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_u32(42);
    /// assert_eq!(big_int.bit_length(), 6); // 42 in binary is 101_010, which has 6 bits
    /// let big_int_zero = BigInteger::with_u32(0);
    pub fn bit_length(&self) -> usize {
        *self.bit_length.get_or_init(|| {
            if self.sign == 0 || self.magnitude.is_empty() {
                0usize
            } else {
                calc_bit_length(self.sign, &self.magnitude)
            }
        })
    }
    pub fn get_length_of_u32_vec(&self) -> usize {
        Self::get_length_u32(self.bit_length())
    }
    pub fn get_length_of_u32_vec_unsigned(&self) -> usize {
        Self::get_length_u32(if self.sign < 0 { self.bit_length() + 1 } else { self.bit_length() })
    }
    /// Returns the number of set bits (ones) in the `BigInteger`.
    ///
    /// This method calculates and caches the count of bits set to 1 in the magnitude.
    /// For negative numbers, it returns the bit count of the bitwise NOT of the value.
    ///
    /// # Returns
    /// * `usize` - The number of set bits in the value.
    ///
    /// # Example
    /// ```rust
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_u32(42);
    /// assert_eq!(big_int.bit_count(), 3); // 42 in binary is 101_010, which has 3 ones
    ///
    /// let big_int_neg = BigInteger::with_i32(-42);
    /// assert_eq!(big_int_neg.bit_count(), 3); // -42 in binary (two's complement) has 3 ones
    ///
    /// let big_int_zero = BigInteger::with_u32(0);
    /// assert_eq!(big_int_zero.bit_count(), 0); // 0 has no ones
    /// ```
    pub fn bit_count(&self) -> usize {
        *self.bits.get_or_init(|| {
            if self.sign < 0 {
                self.not().bit_count()
            } else {
                let mut sum = 0usize;
                for &value in &self.magnitude {
                    sum += value.count_ones() as usize;
                }
                sum
            }
        })
    }
    /// Returns a new `BigInteger` with the sign negated.
    ///
    /// # Example
    /// ```rust
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_u32(42);
    /// let negated = big_int.negate();
    /// assert_eq!(negated.as_i32(), -42);
    ///
    /// let big_int_neg = BigInteger::with_i32(-42);
    /// let negated_neg = big_int_neg.negate();
    /// assert_eq!(negated_neg.as_i32(), 42);
    ///
    /// let big_int_zero = BigInteger::with_u32(0);
    /// let negated_zero = big_int_zero.negate();
    /// assert_eq!(negated_zero.as_i32(), 0);
    /// ```
    pub fn negate(&self) -> Self {
        if self.sign == 0 {
            self.clone()
        } else {
            Self::new(-self.sign, self.magnitude.clone())
        }
    }
    /// Returns the absolute value of the `BigInteger`.
    ///
    /// # Example
    /// ```
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_i32(-42);
    /// let abs_val = big_int.abs();
    /// assert_eq!(abs_val.as_i32(), 42);
    /// ```
    pub fn abs(&self) -> Self {
        if self.sign >= 0 { self.clone() } else { self.negate() }
    }
    /// Increments the value of the `BigInteger` by one and returns the result as a new `BigInteger`.
    ///
    /// If the value is zero, returns a new `BigInteger` representing one.
    /// If the value is negative, decrements the magnitude.
    /// If the value is positive, increments the magnitude.
    ///
    /// # Returns
    /// A new `BigInteger` instance with the incremented value.
    ///
    /// # Example
    /// ```rust
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_u32(42);
    /// let incremented = big_int.inc();
    /// assert_eq!(incremented.as_u32(), 43);
    ///
    /// let big_int_neg = BigInteger::with_i32(-42);
    /// let incremented_neg = big_int_neg.inc();
    /// assert_eq!(incremented_neg.as_i32(), -41);
    /// ```
    pub fn inc(&self) -> Self {
        if self.sign == 0 {
            return Self::with_u8(1);
        }
        if self.sign < 0 {
            Self::with_check_magnitude(-1, do_sub_big_lil(&self.magnitude, &(*ONE).magnitude), true)
        } else {
            self.add_to_magnitude(&(*ONE).magnitude)
        }
    }
    /// Returns the bitwise NOT of the current `BigInteger` (two's complement).
    ///
    /// # Returns
    /// A new `BigInteger` representing the bitwise NOT of the current value.
    ///
    /// # Example
    /// ```rust
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_u32(42);
    /// let not_value = big_int.not();
    /// assert_eq!(not_value.as_i32(), -43); // Bitwise NOT of 42 in two's complement
    ///
    /// let big_int_neg = BigInteger::with_i32(-42);
    /// let not_neg_value = big_int_neg.not();
    /// assert_eq!(not_neg_value.as_i32(), 41); // Bitwise NOT of -42 in two's complement
    ///
    /// let big_int_zero = BigInteger::with_u32(0);
    /// let not_zero_value = big_int_zero.not();
    /// assert_eq!(not_zero_value.as_i32(), -1); // Bitwise NOT of 0 in two's complement
    /// ```
    pub fn not(&self) -> Self {
        let r1 = self.inc();
        let r2 = r1.negate();
        r2
    }
    /// Adds two `BigInteger` values and returns the result as a new `BigInteger`.
    ///
    /// # Arguments
    /// * `other` - A reference to another `BigInteger` to add.
    ///
    /// # Returns
    /// A new `BigInteger` representing the sum of `self` and `other`.
    ///
    /// # Example
    /// ```rust
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int1 = BigInteger::with_u32(42);
    /// let big_int2 = BigInteger::with_u32(58);
    /// let sum = big_int1.add(&big_int2);
    /// assert_eq!(sum.as_u32(), 100); // 42 + 58 = 100
    ///
    /// let big_int_neg1 = BigInteger::with_i32(-42);
    /// let big_int_neg2 = BigInteger::with_i32(-58);
    /// let sum_neg = big_int_neg1.add(&big_int_neg2);
    /// assert_eq!(sum_neg.as_i32(), -100); // -42 + -58 = -100
    ///
    /// let big_int_zero = BigInteger::with_u32(0);
    /// let sum_zero = big_int1.add(&big_int_zero);
    /// assert_eq!(sum_zero.as_u32(), 42); // 42 + 0 = 42
    /// ```
    pub fn add(&self, other: &Self) -> Self {
        if self.sign == 0 {
            other.clone()
        } else if other.sign == 0 {
            self.clone()
        } else if self.sign == other.sign {
            self.add_to_magnitude(&other.magnitude)
        } else if other.sign < 0 {
            self.subtract(&other.negate())
        } else {
            other.subtract(&self.negate())
        }
    }
    /// Subtracts another `BigInteger` from this one and returns the result as a new `BigInteger`.
    ///
    /// # Arguments
    /// * `other` - A reference to the `BigInteger` to subtract.
    ///
    /// # Returns
    /// A new `BigInteger` representing the result of the subtraction.
    ///
    /// # Example
    /// ```
    /// use bc_rust::math::BigInteger;
    ///
    /// let a = BigInteger::with_u32(100);
    /// let b = BigInteger::with_u32(42);
    /// let result = a.subtract(&b);
    /// assert_eq!(result.as_u32(), 58);
    /// ```
    pub fn subtract(&self, other: &Self) -> Self {
        if other.sign == 0 {
            return self.clone();
        } else if self.sign == 0 {
            return other.negate();
        } else if self.sign != other.sign {
            return self.add(&other.negate());
        }
        let compare = compare_no_leading_zeros(self.magnitude.as_slice(), other.magnitude.as_slice());
        if compare == 0 {
            return (*ZERO).clone();
        }
        let (bi, li) = if compare < 0 { (other, self) } else { (self, other) };
        let magnitude = do_sub_big_lil(bi.magnitude.as_slice(), li.magnitude.as_slice());
        Self::with_check_magnitude(self.sign * (compare as i8), magnitude, true)
    }
    /// Multiplies two `BigInteger` values and returns the result as a new `BigInteger`.
    ///
    /// # Arguments
    /// * `other` - A reference to another `BigInteger` to multiply with.
    ///
    /// # Returns
    /// A new `BigInteger` representing the product of `self` and `other`.
    ///
    /// # Example
    /// ```
    /// use bc_rust::math::BigInteger;
    ///
    /// let a = BigInteger::with_u32(6);
    /// let b = BigInteger::with_u32(7);
    /// let product = a.multiply(&b);
    /// assert_eq!(product.as_u32(), 42);
    /// ```
    pub fn multiply(&self, other: &Self) -> Self {
        if self == other {
            return self.square();
        }

        if (self.sign & other.sign) == 0 {
            return (*ZERO).clone();
        }

        if other.quick_pow2_check() {
            let result = self.shift_left((other.abs().bit_length() - 1) as isize);
            return if other.sign > 0 { result } else { result.negate() };
        }

        if self.quick_pow2_check() {
            let result = other.shift_left((self.abs().bit_length() - 1) as isize);
            return if self.sign > 0 { result } else { result.negate() };
        }
        let res_length = self.magnitude.len() + other.magnitude.len();
        let mut res = vec![0u32; res_length];
        multiply(&mut res, &self.magnitude, &other.magnitude);
        let res_sign = self.sign ^ other.sign ^ 1;
        Self::with_check_magnitude(res_sign, res, true)
    }
    /// Divides this `BigInteger` by another `BigInteger` and returns the result as a `Result`.
    ///
    /// # Arguments
    /// * `other` - The divisor as a reference to another `BigInteger`.
    ///
    /// # Returns
    /// * `Result<Self>` - The quotient as a new `BigInteger` if successful, or an error if division by zero occurs.
    ///
    /// # Errors
    /// Returns `BcError` if `other` is zero.
    ///
    /// # Example
    /// ```rust
    /// use bc_rust::math::BigInteger;
    ///
    /// let a = BigInteger::with_u32(100);
    /// let b = BigInteger::with_u32(20);
    /// let result = a.divide(&b);
    /// assert_eq!(result.unwrap().as_u32(), 5); // 100 / 20 = 5
    ///
    /// let a_neg = BigInteger::with_i32(-100);
    /// let b_neg = BigInteger::with_i32(-20);
    /// let result_neg = a_neg.divide(&b_neg);
    /// assert_eq!(result_neg.unwrap().as_i32(), 5); // -100 / -20 = 5
    ///
    /// let a_zero = BigInteger::with_u32(0);
    /// let b_zero = BigInteger::with_u32(20);
    /// let result_zero = a_zero.divide(&b_zero);
    /// assert_eq!(result_zero.unwrap().as_u32(), 0); // 0 / 20 = 0
    ///
    /// let a_neg_zero = BigInteger::with_i32(-0);
    /// let b_neg_zero = BigInteger::with_i32(-20);
    /// let result_neg_zero = a_neg_zero.divide(&b_neg_zero);
    /// assert_eq!(result_neg_zero.unwrap().as_i32(), 0); // -0 / -20 = 0
    ///
    /// let a_div_zero = BigInteger::with_u32(100);
    /// let b_div_zero = BigInteger::with_u32(0);
    /// let result_div_zero = a_div_zero.divide(&b_div_zero);
    /// assert!(result_div_zero.is_err()); // Division by zero should return an error
    /// ```
    pub fn divide(&self, other: &Self) -> Result<Self> {
        if other.sign == 0 || other.magnitude.is_empty() {
            return Err(BcError::with_invalid_argument("divide by zero"));
        }

        if self.sign == 0 {
            return Ok(BigInteger::with_u32(0));
        }

        if other.quick_pow2_check() {
            let result = self.abs().shift_right((other.abs().bit_length() - 1) as isize);
            return if other.sign == self.sign { Ok(result) } else { Ok(result.negate()) };
        }

        let mut mag = self.magnitude.to_vec();
        Ok(Self::with_check_magnitude(
            self.sign * other.sign,
            divide(&mut mag, &other.magnitude),
            true,
        ))
    }
    /// Returns the remainder of division of `self` by `division` as a `Result`.
    ///
    /// # Arguments
    /// * `division` - The divisor as a reference to another `BigInteger`.
    ///
    /// # Returns
    /// * `Result<Self>` - The remainder as a new `BigInteger` if successful, or an error if division by zero occurs.
    ///
    /// # Errors
    /// Returns `BcError` if `division` is zero.
    ///
    /// # Example
    /// ```rust
    /// use bc_rust::math::BigInteger;
    ///
    /// let a = BigInteger::with_u32(100);
    /// let b = BigInteger::with_u32(30);
    /// let result = a.remainder(&b);
    /// assert_eq!(result.unwrap().as_u32(), 10); // 100 % 30 = 10
    ///
    /// let a_neg = BigInteger::with_i32(-100);
    /// let b_neg = BigInteger::with_i32(30);
    /// let result_neg = a_neg.remainder(&b_neg);
    /// assert_eq!(result_neg.unwrap().as_i32(), -10); // -100 % 30 = -10
    ///
    /// ```
    pub fn remainder(&self, division: &Self) -> Result<Self> {
        if division.sign == 0 || division.magnitude.is_empty() {
            return Err(BcError::with_invalid_argument("divide by zero"));
        }

        if self.sign == 0 {
            return Ok(BigInteger::with_u32(0));
        }

        // For small values, use fast remainder method
        if division.magnitude.len() == 1 {
            let val = division.magnitude[0];
            if val > 0 {
                if val == 1 {
                    return Ok(BigInteger::with_u32(0));
                }
                let rem = self.remainder_with_u32(val);
                return if rem == 0 {
                    Ok(BigInteger::with_u32(0))
                } else {
                    Ok(BigInteger::with_check_magnitude(self.sign, vec![rem], false))
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
            result = self.magnitude.clone();
            remainder(&mut result, &division.magnitude);
        }

        Ok(Self::with_check_magnitude(self.sign, result, true))
    }
    /// Divides `self` by `other` and returns a tuple containing the quotient and remainder as `BigInteger`.
    ///
    /// # Arguments
    /// * `other` - The divisor as a reference to another `BigInteger`.
    ///
    /// # Returns
    /// * `Result<(BigInteger, BigInteger)>` - On success, returns a tuple `(quotient, remainder)`. Returns an error if division by zero occurs.
    ///
    /// # Errors
    /// Returns `BcError` if `other` is zero.
    ///
    /// # Example
    /// ```
    /// use bc_rust::math::BigInteger;
    ///
    /// let a = BigInteger::with_u32(100);
    /// let b = BigInteger::with_u32(20);
    /// let (q, r) = a.divide_and_remainder(&b).unwrap();
    /// assert_eq!(q.as_u32(), 5);
    /// assert_eq!(r.as_u32(), 0);
    /// ```
    pub fn divide_and_remainder(&self, other: &Self) -> Result<(Self, Self)> {
        if other.sign == 0 || other.magnitude.is_empty() {
            return Err(BcError::with_invalid_argument("divide by zero"));
        }

        if self.sign == 0 {
            return Ok((BigInteger::with_u32(0), BigInteger::with_u32(0)));
        }

        if other.quick_pow2_check() {
            let e = other.abs().bit_length() - 1;
            let quotient = self.abs().shift_right(e as isize);

            let divide = if self.sign == other.sign { quotient } else { quotient.negate() };

            let remainder = Self::with_check_magnitude(self.sign, self.last_n_bits(e), true);
            return Ok((divide, remainder));
        }

        let mut remainder = self.magnitude.clone();
        let quotient = divide(&mut remainder, &other.magnitude);
        Ok((
            Self::with_check_magnitude(self.sign * other.sign, quotient, true),
            Self::with_check_magnitude(self.sign, remainder, true),
        ))
    }
    /// Calculates the mathematical modulus of `self` with respect to `other`.
    ///
    /// This returns a non-negative result, even if `self` is negative.
    /// If the result of `self.remainder(other)` is negative, it adds `other` to ensure the result is positive.
    ///
    /// # Arguments
    /// * `other` - The divisor as a reference to another `BigInteger`.
    ///
    /// # Returns
    /// * `Result<Self>` - The modulus as a new `BigInteger` if successful, or an error if division by zero occurs.
    ///
    /// # Errors
    /// Returns `BcError` if `other` is zero.
    ///
    /// # Example
    /// ```
    /// use bc_rust::math::BigInteger;
    ///
    /// let a = BigInteger::with_i32(-13);
    /// let b = BigInteger::with_i32(5);
    /// let result = a.modulus(&b);
    /// assert_eq!(result.unwrap().as_i32(), 2); // -13 mod 5 = 2
    /// ```
    pub fn modulus(&self, other: &Self) -> Result<Self> {
        if other.sign == 0 || other.magnitude.is_empty() {
            return Err(BcError::with_invalid_argument("divide by zero"));
        }

        let biggie = self.remainder(other)?;
        if biggie.sign >= 0 { Ok(biggie) } else { Ok(biggie.add(other)) }
    }
    pub fn mod_inverse(&self, modulus: &Self) -> Result<Self> {
        if self.sign <= 0 {
            return Err(BcError::with_invalid_argument("sModulus must be positive"));
        }

        if modulus.quick_pow2_check() {
            return Ok(self.mod_inverse_pow2(modulus)?);
        }

        let d = self.remainder(modulus)?;
        let (gcd, mut x) = ext_euclid(&d, modulus);

        if gcd != *ONE {
            return Err(BcError::with_arithmetic_error("Numbers not relatively prime"));
        }

        if x.sign < 0 {
            x = x.add(modulus);
        }
        Ok(x)
    }
    pub fn mod_pow(&self, e: &Self, m: &Self) -> Result<Self> {
        if m.sign <= 0 {
            return Err(BcError::with_invalid_argument("Modulus must be positive"));
        }

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

        let mut result = self.modulus(m)?;
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
    /// Calculates the greatest common divisor (GCD) of `self` and `other`.
    ///
    /// # Arguments
    /// * `other` - A reference to another `BigInteger` to compute the GCD with.
    ///
    /// # Returns
    /// * `Result<Self>` - The GCD as a new `BigInteger`.
    ///
    /// # Example
    /// ```
    /// use bc_rust::math::BigInteger;
    ///
    /// let a = BigInteger::with_u32(48);
    /// let b = BigInteger::with_u32(18);
    /// let result = a.gcd(&b);
    /// assert_eq!(result.unwrap().as_u32(), 6);
    /// ```
    pub fn gcd(&self, other: &Self) -> Result<Self> {
        if other.sign == 0 || other.magnitude.is_empty() {
            return Ok(self.abs());
        }

        if self.sign == 0 {
            return Ok(other.abs());
        }

        let mut r: BigInteger;
        let mut u = self.clone();
        let mut v = other.clone();
        while v.sign != 0 {
            r = u.modulus(&v)?;
            u = v;
            v = r;
        }
        Ok(u)
    }
    /// Returns the square of this `BigInteger`.
    ///
    /// # Returns
    /// A new `BigInteger` representing the value of `self * self`.
    ///
    /// # Example
    /// ```
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_u32(7);
    /// let squared = big_int.square();
    /// assert_eq!(squared.as_u32(), 49);
    /// ```
    pub fn square(&self) -> Self {
        if self.sign == 0 {
            return BigInteger::with_u32(0);
        }

        if self.quick_pow2_check() {
            return self.shift_left((self.abs().bit_length() - 1) as isize);
        }

        let mut res_length = self.magnitude.len() << 1;
        if self.magnitude[0] >> 16 == 0 {
            res_length -= 1;
        }
        let mut res = vec![0u32; res_length];
        square(&mut res, &self.magnitude);
        Self::with_check_magnitude(1, res, false)
    }
    pub fn pow(&self, exp: u32) -> Result<Self> {
        let mut exp = exp;
        if exp == 0 {
            return Ok((*ONE).clone());
        }
        if self.sign == 0 {
            return Ok(self.clone());
        }
        if self.quick_pow2_check() {
            let pow_of_2 = exp as u64 * (self.bit_length() - 1) as u64;

            if pow_of_2 > i32::MAX as u64 {
                return Err(BcError::with_arithmetic_error("power of 2"));
            }

            return Ok((*ONE).shift_left(pow_of_2 as isize));
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
    /// Performs a bitwise AND operation between this `BigInteger` and another.
    ///
    /// # Arguments
    /// * `other` - A reference to another `BigInteger` to AND with.
    ///
    /// # Returns
    /// A new `BigInteger` representing the result of the bitwise AND.
    ///
    /// # Example
    /// ```rust
    /// use bc_rust::math::BigInteger;
    ///
    /// let a = BigInteger::with_u32(42); // 101010 in binary
    /// let b = BigInteger::with_u32(58); // 111010 in binary
    /// let result = a.and(&b);
    /// assert_eq!(result.as_u32(), 42); // 101_010 AND 111_010 = 101_010
    ///
    /// let a_neg = BigInteger::with_i32(-42); // 101_010 in binary (two's complement)
    /// let b_neg = BigInteger::with_i32(-58); // 111_010 in binary (two's complement)
    /// let result_neg = a_neg.and(&b_neg);
    /// assert_eq!(result_neg.as_i32(), -58); // Bitwise AND of -42 and -58
    ///
    /// let a_zero = BigInteger::with_u32(0);
    /// let b_zero = BigInteger::with_u32(0);
    /// let result_zero = a_zero.and(&b_zero);
    /// assert_eq!(result_zero.as_u32(), 0); // 0 AND 0 = 0
    ///
    /// let a_one = BigInteger::with_u32(42);
    /// let b_one = BigInteger::with_u32(0);
    /// let result_one = a_one.and(&b_one);
    /// assert_eq!(result_one.as_u32(), 0); // 42 AND 0 = 0
    ///
    /// let a_neg_one = BigInteger::with_i32(-42);
    /// let b_neg_one = BigInteger::with_i32(0);
    /// let result_neg_one = a_neg_one.and(&b_neg_one);
    /// assert_eq!(result_neg_one.as_i32(), 0); // -42 AND 0 = 0
    /// ```
    pub fn and(&self, other: &Self) -> Self {
        if self.sign == 0 || other.sign == 0 {
            return BigInteger::with_u32(0);
        }

        let a_magnitude = if self.sign > 0 {
            self.magnitude.clone()
        } else {
            self.add(&(*ONE)).magnitude.clone()
        };

        let b_magnitude = if other.sign > 0 {
            other.magnitude.clone()
        } else {
            other.add(&(*ONE)).magnitude.clone()
        };

        let result_neg = self.sign < 0 && other.sign < 0;
        let result_length = std::cmp::max(a_magnitude.len(), b_magnitude.len());
        let mut result_mag = vec![0u32; result_length];
        let a_start = (result_mag.len() - a_magnitude.len()) as isize;
        let b_start = (result_mag.len() - b_magnitude.len()) as isize;
        for i in 0..result_mag.len() {
            let mut a_word = if (i as isize) >= a_start {
                a_magnitude[i - a_start as usize]
            } else {
                0u32
            };
            let mut b_word = if (i as isize) >= b_start {
                b_magnitude[i - b_start as usize]
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

        let mut result = Self::with_check_magnitude(1, result_mag, true);
        if result_neg {
            result = result.not()
        }
        result
    }
    /// Performs a bitwise OR operation between this `BigInteger` and another.
    ///
    /// # Arguments
    /// * `other` - A reference to another `BigInteger` to OR with.
    ///
    /// # Returns
    /// A new `BigInteger` representing the result of the bitwise OR.
    ///
    /// # Example
    /// ```rust
    /// use bc_rust::math::BigInteger;
    ///
    /// let a = BigInteger::with_u32(42); // 101010 in binary
    /// let b = BigInteger::with_u32(58); // 111010 in binary
    /// let result = a.or(&b);
    /// assert_eq!(result.as_u32(), 58); // 101010 OR 111010 = 111010
    /// ```
    pub fn or(&self, other: &Self) -> Self {
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
            let mut a_word = if (i as isize) >= a_start { a_mag[i - a_start as usize] } else { 0u32 };
            let mut b_word = if (i as isize) >= b_start { b_mag[i - b_start as usize] } else { 0u32 };
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
        let mut result = BigInteger::with_check_magnitude(1, result_mag, true);
        if result_neg {
            result = result.not();
        }
        result
    }
    /// Performs a bitwise AND-NOT operation between this `BigInteger` and another.
    ///
    /// # Arguments
    /// * `other` - A reference to another `BigInteger` to apply NOT, then AND with `self`.
    ///
    /// # Returns
    /// A new `BigInteger` representing the result of `self & !other`.
    ///
    /// # Example
    /// ```rust
    /// use bc_rust::math::BigInteger;
    ///
    /// let a = BigInteger::with_u32(42); // 101010 in binary
    /// let b = BigInteger::with_u32(58); // 111010 in binary
    /// let result = a.and_not(&b);
    /// assert_eq!(result.as_u32(), 0); // 101010 AND NOT 111_010 = 101_010 AND 000101 = 000000
    pub fn and_not(&self, other: &Self) -> Self {
        let r1 = other.not();
        let r2 = self.and(&r1);
        r2
    }
    /// Performs a bitwise XOR operation between this `BigInteger` and another.
    ///
    /// # Arguments
    /// * `value` - A reference to another `BigInteger` to XOR with.
    ///
    /// # Returns
    /// A new `BigInteger` representing the result of the bitwise XOR.
    ///
    /// # Example
    /// ```rust
    /// use bc_rust::math::BigInteger;
    ///
    /// let a = BigInteger::with_u32(42); // 101010 in binary
    /// let b = BigInteger::with_u32(58); // 111010 in binary
    /// let result = a.xor(&b);
    /// assert_eq!(result.as_u32(), 16); // 101010 XOR 111010 = 010000
    /// ```
    pub fn xor(&self, value: &Self) -> Self {
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

        let result_neg = (self.sign < 0 && value.sign >= 0) || (self.sign >= 0 && value.sign < 0);
        let result_length = std::cmp::max(a_mag.len(), b_mag.len());
        let mut result_mag = vec![0u32; result_length];

        let a_start = result_mag.len() - a_mag.len();
        let b_start = result_mag.len() - b_mag.len();

        for i in 0..result_mag.len() {
            let mut a_word = if i >= a_start { a_mag[i - a_start] } else { 0u32 };
            let mut b_word = if i >= b_start { b_mag[i - b_start] } else { 0u32 };

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

        let mut result = BigInteger::with_check_magnitude(1, result_mag, true);
        if result_neg {
            result = result.not();
        }
        result
    }
    /// Tests whether the bit at position `n` is set in the `BigInteger`.
    ///
    /// For negative numbers, this returns the logical NOT of the corresponding bit in the two's complement representation.
    ///
    /// # Arguments
    /// * `n` - The zero-based position of the bit to test.
    ///
    /// # Returns
    /// * `true` if the bit at position `n` is set, `false` otherwise.
    ///
    /// # Example
    /// ```
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_u32(5); // 0b101
    /// assert!(big_int.test_bit(0)); // true, the least significant bit is set
    /// assert!(!big_int.test_bit(1)); // false
    /// assert!(big_int.test_bit(2)); // true
    /// ```
    pub fn test_bit(&self, n: usize) -> bool {
        if self.sign < 0 {
            return !self.not().test_bit(n);
        }

        let word_num = n / 32;
        if word_num >= self.magnitude.len() {
            return false;
        }
        let word = self.magnitude[self.magnitude.len() - 1 - word_num];
        ((word >> (n % 32)) & 1) != 0
    }
    /// Sets the bit at position `n` in the `BigInteger`.
    ///
    /// # Arguments
    /// * `n` - The zero-based position of the bit to set.
    ///
    /// # Returns
    /// A new `BigInteger` with the bit at position `n` set to 1.
    ///
    /// # Example
    /// ```
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_u32(5); // 0b101
    /// let new_big_int = big_int.set_bit(1); // Sets the second bit
    /// assert_eq!(new_big_int.as_u32(), 7); // 0b111
    /// ```
    pub fn set_bit(&self, n: usize) -> Self {
        if self.test_bit(n) {
            return self.clone();
        }
        if self.sign > 0 && n < (self.bit_length() - 1) {
            self.flip_existing_bit(n)
        } else {
            self.or(&(*ONE).shift_left(n as isize))
        }
    }
    /// Clears the bit at position `n` in the `BigInteger`.
    ///
    /// # Arguments
    /// * `n` - The zero-based position of the bit to clear.
    ///
    /// # Returns
    /// A new `BigInteger` with the bit at position `n` cleared to 0.
    ///
    /// # Example
    /// ```
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_u32(7); // 0b111
    /// let new_big_int = big_int.clear_bit(1); // Clears the second bit
    /// assert_eq!(new_big_int.as_u32(), 5); // 0b101
    /// ```
    pub fn clear_bit(&self, n: usize) -> Self {
        if !self.test_bit(n) {
            return self.clone();
        }

        if self.sign > 0 && n < (self.bit_length() - 1) {
            return self.flip_existing_bit(n);
        }

        let r1 = (*ONE).shift_left(n as isize);
        let r2 = self.and_not(&r1);
        r2
    }
    pub fn flip_bit(&self, n: usize) -> Self {
        if self.sign > 0 && n < (self.bit_length() - 1) {
            return self.flip_existing_bit(n);
        }
        let n1 = &(*ONE).shift_left(n as isize);
        self.xor(n1)
    }
    pub fn get_lowest_set_bit(&self) -> i32 {
        if self.sign == 0 {
            return -1;
        }
        self.get_lowest_set_bit_mask_first(u32::MAX)
    }
    /// Performs a left bitwise shift on the `BigInteger` by `n` bits.
    ///
    /// # Arguments
    /// * `n` - The number of bits to shift left. If negative, performs a right shift.
    ///
    /// # Returns
    /// A new `BigInteger` with the value shifted left by `n` bits.
    ///
    /// # Example
    /// ```
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_u32(1);
    /// let shifted = big_int.shift_left(3);
    /// assert_eq!(shifted.as_u32(), 8); // 1 << 3 = 8
    ///
    /// let big_int_neg = BigInteger::with_i32(-1);
    /// let shifted_neg = big_int_neg.shift_left(3);
    /// assert_eq!(shifted_neg.as_i32(), -8); // -1 << 3 = -8
    /// ```
    pub fn shift_left(&self, n: isize) -> Self {
        if self.sign == 0 || self.magnitude.is_empty() {
            return BigInteger::with_u32(0);
        }
        if n == 0 {
            return self.clone();
        }
        if n < 0 {
            return self.shift_right(-n);
        }

        let result = Self::with_check_magnitude(self.sign, shift_left(&self.magnitude, n as usize), true);
        result.bits.get_or_init(|| {
            if result.sign > 0 {
                self.bit_count()
            } else {
                self.bit_count() + (n as usize)
            }
        });

        result.bit_length.get_or_init(|| self.bit_length() + (n as usize));

        result
    }
    /// Performs a right bitwise shift on the `BigInteger` by `n` bits.
    ///
    /// # Arguments
    /// * `n` - The number of bits to shift right. If negative, performs a left shift.
    ///
    /// # Returns
    /// A new `BigInteger` with the value shifted right by `n` bits.
    ///
    /// # Example
    /// ```
    /// use bc_rust::math::BigInteger;
    ///
    /// let big_int = BigInteger::with_u32(8);
    /// let shifted = big_int.shift_right(3);
    /// assert_eq!(shifted.as_u32(), 1); // 8 >> 3 = 1
    ///
    /// let big_int_neg = BigInteger::with_i32(-8);
    /// let shifted_neg = big_int_neg.shift_right(3);
    /// assert_eq!(shifted_neg.as_i32(), -1); // -8 >> 3 = -1
    /// ```
    pub fn shift_right(&self, n: isize) -> Self {
        if n == 0 {
            return self.clone();
        }

        if n < 0 {
            return self.shift_left(-n);
        }

        if n as usize >= self.bit_length() {
            return if self.sign < 0 { (*ONE).negate() } else { (*ZERO).clone() };
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
        Self::with_check_magnitude(self.sign, res, false)
    }
    pub fn is_probable_prime(&self, certainty: i32) -> Result<bool> {
        self.is_probable_prime_with_randomly_selected(certainty, false)
    }
    pub fn next_probable_prime(&self) -> Result<Self> {
        if self.sign < 0 {
            return Err(BcError::with_invalid_argument("Negative numbers cannot be prime"));
        }

        if self < &(*TWO) {
            return Ok((*TWO).clone());
        }

        let mut n = self.inc().set_bit(0);
        while !n.check_probable_prime(100, &mut rand::rng(), false)? {
            n = n.add(&(*TWO));
        }
        Ok(n)
    }
    pub fn max(&self, value: &Self) -> Self {
        if self < value { value.clone() } else { self.clone() }
    }
    pub fn min(&self, value: &Self) -> Self {
        if self < value { self.clone() } else { value.clone() }
    }
    pub(crate) fn is_probable_prime_with_randomly_selected(&self, certainty: i32, randomly_selected: bool) -> Result<bool> {
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

        Ok(n.check_probable_prime(certainty, &mut rand::rng(), randomly_selected)?)
    }

    fn add_to_magnitude(&self, other: &[u32]) -> Self {
        let (big, small): (&[u32], &[u32]) = if self.magnitude.len() < other.len() {
            (other, &self.magnitude)
        } else {
            (&self.magnitude, other)
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
        Self::with_check_magnitude(self.sign, big_copy, possible_over_flow)
    }
    fn quick_pow2_check(&self) -> bool {
        self.sign > 0 && self.bit_count() == 1usize
    }
    fn last_n_bits(&self, n: usize) -> Vec<u32> {
        if n == 0 {
            return vec![];
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
    fn remainder_with_u32(&self, m: u32) -> u32 {
        debug_assert!(m > 0);
        let mut acc = 0u64;
        for i in 0..self.magnitude.len() {
            let pos_val = self.magnitude[i];
            acc = (acc << 32 | pos_val as u64) % m as u64;
        }
        acc as u32
    }
    fn check_probable_prime(&self, certainty: i32, random: &mut dyn RngCore, randomly_selected: bool) -> Result<bool> {
        debug_assert!(certainty > 0);
        debug_assert!(self > &(*TWO));
        debug_assert!(self.test_bit(0));

        // Try to reduce the penalty for tiny numbers
        let num_lists = std::cmp::min(self.bit_length() - 1, PRIME_LISTS.len());
        for i in 0..num_lists {
            let test = self.remainder_with_u32(*(&(*PRIME_PRODUCTS)[i]));
            let prime_list = &(*PRIME_LISTS)[i];
            for j in 0..prime_list.len() {
                let prime = prime_list[j];
                let q_rem = test % prime;
                if q_rem == 0 {
                    return Ok(self.bit_length() < 16 && self.as_i32() as u32 == prime);
                }
            }
        }
        self.rabin_miller_test_with_randomly_selected(certainty, random, randomly_selected)
    }
    fn rabin_miller_test_with_randomly_selected(&self, certainty: i32, random: &mut dyn RngCore, randomly_selected: bool) -> Result<bool> {
        let bits = self.bit_length();

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
        let r = n.shift_right(s as isize);

        // NOTE: Avoid conversion to/from Montgomery form and check for R/-R as a result instead

        let mont_radix = (*ONE).shift_left((32 * n.magnitude.len()) as isize).remainder(&n)?;
        let minus_mont_radix = n.subtract(&mont_radix);

        let mut y_accum = vec![0u32; n.magnitude.len() + 1];

        loop {
            let mut a: BigInteger;
            loop {
                a = Self::with_random(n.bit_length(), random);
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
        Ok(true)
    }
    fn mod_square_monty(y_accum: &mut [u32], b: &Self, m: &Self) -> Self {
        let n = m.magnitude.len();
        let pow_r = 32 * n;
        let small_monty_modulus = m.bit_length() + 2 <= pow_r;
        let m_dash = m.get_m_quote();

        debug_assert!(y_accum.len() == n + 1);

        let z_val = b.magnitude.to_vec();
        debug_assert!(z_val.len() <= n);

        let mut y_val = vec![0u32; n];
        y_val[n - z_val.len()..].copy_from_slice(&z_val);

        square_monty(y_accum, &mut y_val, &m.magnitude, m_dash, small_monty_modulus);

        if small_monty_modulus && compare_to(&y_val, &m.magnitude) >= 0 {
            subtract(&mut y_val, &m.magnitude);
        }

        Self::with_check_magnitude(1, y_val, true)
    }
    /// Calculate mQuote = -m^(-1) mod b with b = 2^32 (32 = word size)
    fn get_m_quote(&self) -> u32 {
        debug_assert!(self.sign > 0);
        let d = 0u32.wrapping_sub(self.magnitude[self.magnitude.len() - 1]);
        debug_assert!((d & 1) != 0);
        inverse_u32(d)
    }
    fn mod_pow_monty(y_acc_m: &mut [u32], b: &Self, e: &Self, m: &Self, convert: bool) -> Result<Self> {
        let n = m.magnitude.len();
        let pow_r = 32 * n;
        let small_monty_modulus = m.bit_length() + 2 <= pow_r;
        let m_dash = m.get_m_quote();

        // tmp = this * R mod m
        let mut b1 = b.clone();
        if convert {
            b1 = b1.shift_left(pow_r as isize).remainder(m)?;
        }
        debug_assert!(y_acc_m.len() == n + 1);

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
        if e.magnitude.len() > 1 || e.bit_count() > 2 {
            let exp_length = e.bit_length();
            while exp_length > EXP_WINDOW_THRESHOLDS[extra_bits] {
                extra_bits += 1;
            }
        }

        let num_powers = 1usize << extra_bits;
        let mut odd_powers = vec![vec![0u32; 0]; num_powers];
        odd_powers[0] = z_val.clone();

        let mut z_squared = z_val.clone();
        square_monty(y_acc_m, &mut z_squared, &m.magnitude, m_dash, small_monty_modulus);

        for i in 1..num_powers {
            odd_powers[i] = odd_powers[i - 1].clone();
            multiply_monty(y_acc_m, &mut odd_powers[i], &z_squared, &m.magnitude, m_dash, small_monty_modulus);
        }

        let window_list = get_window_list(&e.magnitude, extra_bits);
        debug_assert!(window_list.len() > 1);

        let mut window = window_list[0];
        let mut mul_t = window & 0xFF;
        let mut last_zeros = window >> 8;

        let mut y_val: Vec<u32>;
        if mul_t == 1 {
            y_val = z_squared;
            last_zeros = last_zeros.wrapping_sub(1);
        } else {
            y_val = odd_powers[(mul_t >> 1) as usize].clone();
        }
        let mut window_pos = 1;
        while {
            window = window_list[window_pos];
            window_pos += 1;
            window
        } != u32::MAX
        {
            mul_t = window & 0xFF;
            let bits = last_zeros as i32 + bit_len(mul_t) as i32;
            for _ in 0..bits {
                square_monty(y_acc_m, &mut y_val, &m.magnitude, m_dash, small_monty_modulus);
            }

            multiply_monty(
                y_acc_m,
                &mut y_val,
                &odd_powers[(mul_t >> 1) as usize],
                &m.magnitude,
                m_dash,
                small_monty_modulus,
            );

            last_zeros = window >> 8;
        }

        for _ in 0..last_zeros {
            square_monty(y_acc_m, &mut y_val, &m.magnitude, m_dash, small_monty_modulus);
        }

        if convert {
            // Return y * R^(-1) mod m
            montgomery_reduce(&mut y_val, &m.magnitude, m_dash);
        } else if small_monty_modulus && compare_to(&y_val, &m.magnitude) >= 0 {
            subtract(&mut y_val, &m.magnitude);
        }

        Ok(BigInteger::with_check_magnitude(1, y_val, true))
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
        offset
    }
    fn flip_existing_bit(&self, n: usize) -> Self {
        debug_assert!(self.sign > 0);
        debug_assert!(n < self.bit_length() - 1);

        let mut mag = self.magnitude.to_vec();
        let mag_len = mag.len();
        let v = (1 << (n as i32 & 31)) as u32;
        mag[mag_len - 1 - (n as i32 >> 5) as usize] ^= v;
        BigInteger::with_check_magnitude(self.sign, mag, false)
    }
    fn mod_inverse_pow2(&self, m: &Self) -> Result<Self> {
        debug_assert!(m.sign > 0);
        debug_assert!(m.bit_count() == 1);

        if !self.test_bit(0) {
            return Err(BcError::with_arithmetic_error("Numbers not relatively prime"));
        }

        let pow = m.bit_length() << 1;
        let mut inv64 = inverse_u64(self.as_i64() as u64) as i64;
        if pow < 64 {
            inv64 &= (1 << pow) - 1;
        }

        let mut x = BigInteger::with_i64(inv64);

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
    fn mod_pow_barrett(b: &BigInteger, e: &Self, m: &Self) -> Result<Self> {
        let k = m.magnitude.len();
        let mr = (*ONE).shift_left(((k + 1) << 5) as isize);
        let yu = (*ONE).shift_left((k << 6) as isize).divide(m)?;

        // Sliding window from MSW to LSW
        let mut extra_bits = 0;
        let exp_length = e.bit_length();
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
        let mut mul_t = window & 0xFF;
        let mut last_zeros = window >> 8;

        let mut y: BigInteger;
        if mul_t == 1 {
            y = b2.clone();
            last_zeros -= 1;
        } else {
            y = odd_powers[(mul_t >> 1) as usize].clone();
        }

        let mut window_pos = 1;
        while {
            window = window_list[window_pos];
            window_pos += 1;
            window
        } != u32::MAX
        {
            mul_t = window & 0xFF;
            let bits = last_zeros + bit_len(mul_t) as u32;
            for _ in 0..bits {
                y = Self::reduce_barrett(&y.square(), m, &mr, &yu);
            }
            y = Self::reduce_barrett(&(y.multiply(&odd_powers[(mul_t >> 1) as usize])), m, &mr, &yu);
            last_zeros = window >> 8;
        }

        for _ in 0..last_zeros {
            y = Self::reduce_barrett(&(y.square()), m, &mr, &yu);
        }

        Ok(y)
    }
    fn reduce_barrett(x: &Self, m: &Self, mr: &Self, yu: &Self) -> Self {
        let x_len = x.bit_length();
        let m_len = m.bit_length();
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
    fn divide_words(&self, w: u32) -> Self {
        let n = self.magnitude.len();
        if w as usize >= n {
            return (*ZERO).clone();
        }
        let mut mag = vec![0u32; n - w as usize];
        mag.copy_from_slice(&self.magnitude[..(n - w as usize)]);
        BigInteger::with_check_magnitude(self.sign, mag, false)
    }
    fn remainder_words(&self, w: u32) -> Self {
        let n = self.magnitude.len();
        if w as usize >= n {
            return self.clone();
        }
        let mut mag = vec![0u32; w as usize];
        mag.copy_from_slice(&self.magnitude[(n - w as usize)..]);
        BigInteger::with_check_magnitude(self.sign, mag, false)
    }
    fn to_vec_with_signed(&self, unsigned: bool) -> Vec<u8> {
        if self.sign == 0 {
            return if unsigned { vec![0u8; 0] } else { vec![0u8; 1] };
        }
        let n_bits = if unsigned && self.sign > 0 {
            self.bit_length()
        } else {
            self.bit_length() + 1
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
    fn get_length_u32(bits: usize) -> usize {
        bits + size_of::<u32>() - 1 / size_of::<u32>()
    }
    fn copy_to_u32_vec_big_endian(&self, unsigned: bool, output: &mut [u32]) -> Result<()> {
        if self.sign == 0 {
            if !unsigned {
                if output.is_empty() {
                    return Err(BcError::with_invalid_argument("Output vector is empty"));
                }
                output[0] = u32::MIN;
            }
            return Ok(());
        }

        let bits =  if unsigned && self.sign > 0 { self.bit_length() } else { self.bit_length() + 1 };
        let n_u32s = Self::get_length_u32(bits);
        if n_u32s > output.len() {
            return Err(BcError::with_invalid_argument("insufficient space"));
        }

        let mut mag_index = self.magnitude.len();
        let mut ints_index = n_u32s;

        if self.sign > 0 {
            while mag_index > 0 {
                output[{ints_index -= 1; ints_index}] = self.magnitude[{ mag_index -= 1; mag_index }];
            }
            debug_assert!((ints_index & 1) == ints_index);
            if ints_index != 0 {
                output[0] = u32::MIN;
            }
        } else {
            let mut cc = 1u64;
            while mag_index > 0 {
                cc += !self.magnitude[{ mag_index -= 1; mag_index }] as u64;
                output[{ ints_index -= 1; ints_index }] = cc as u32;
                cc >>= 32;
            }
            debug_assert!(cc == 0);
            debug_assert!((ints_index & 1) == ints_index);
            if ints_index != 0 {
                output[{ ints_index -= 1; ints_index}] = u32::MAX;
            }
        }
        Ok(())
    }
    fn copy_to_u32_vec_little_endian(&self, unsigned: bool, output: &mut [u32]) -> Result<()> {
        if self.sign == 0 {
            if !unsigned {
                if output.is_empty() {
                    return Err(BcError::with_invalid_argument("Output vector is empty"));
                }
                output[0] = u32::MIN;
            }
            return Ok(());
        }

        let bits =  if unsigned && self.sign > 0 { self.bit_length() } else { self.bit_length() + 1 };
        let n_u32s = Self::get_length_u32(bits);
        if n_u32s > output.len() {
            return Err(BcError::with_invalid_argument("insufficient space"));
        }

        let mut mag_index = self.magnitude.len();

        if self.sign > 0 {
            for ints_index in 0..self.magnitude.len() {
                output[ints_index] = self.magnitude[{ mag_index -= 1; mag_index }];
            }

            if n_u32s > self.magnitude.len() {
                debug_assert!((self.magnitude.len() + 1) == n_u32s);
                output[self.magnitude.len()] = u32::MIN;
            }
        } else {
            let mut cc = 1u64;
            for ints_index in 0..self.magnitude.len() {
                cc += !self.magnitude[{ mag_index -= 1; mag_index }] as u64;
                output[ints_index] = cc as u32;
                cc >>= 32;
            }
            debug_assert!(cc == 0);

            if n_u32s > self.magnitude.len() {
                debug_assert!((self.magnitude.len() + 1) == n_u32s);
                output[self.magnitude.len()] = u32::MAX;
            }
        }
        Ok(())
    }
}

impl From<u8> for BigInteger {
    fn from(value: u8) -> Self {
        Self::with_u8(value)
    }
}
impl From<u16> for BigInteger {
    fn from(value: u16) -> Self {
        Self::with_u16(value)
    }
}
impl From<u32> for BigInteger {
    fn from(value: u32) -> Self {
        Self::with_u32(value)
    }
}
impl From<u64> for BigInteger {
    fn from(value: u64) -> Self {
        Self::with_u64(value)
    }
}
impl From<u128> for BigInteger {
    fn from(value: u128) -> Self {
        Self::with_u128(value)
    }
}
impl From<i8> for BigInteger {
    fn from(value: i8) -> Self {
        Self::with_i8(value)
    }
}
impl From<i16> for BigInteger {
    fn from(value: i16) -> Self {
        Self::with_i16(value)
    }
}
impl From<i32> for BigInteger {
    fn from(value: i32) -> Self {
        Self::with_i32(value)
    }
}
impl From<i64> for BigInteger {
    fn from(value: i64) -> Self {
        Self::with_i64(value)
    }
}
impl From<i128> for BigInteger {
    fn from(value: i128) -> Self {
        Self::with_i128(value)
    }
}
impl PartialEq for BigInteger {
    fn eq(&self, other: &Self) -> bool {
        self.sign == other.sign && is_equal_magnitude(&self.magnitude, &other.magnitude)
    }
}
impl PartialOrd for BigInteger {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.sign < other.sign {
            Some(Ordering::Less)
        } else if self.sign > other.sign {
            Some(Ordering::Greater)
        } else if self.sign == 0 {
            Some(Ordering::Equal)
        } else {
            let compare = (self.sign as i32) * compare_no_leading_zeros(&self.magnitude, &other.magnitude);
            if compare == 0 {
                Some(Ordering::Equal)
            } else if compare < 0 {
                Some(Ordering::Less)
            } else {
                Some(Ordering::Greater)
            }
        }
    }
}
impl Hash for BigInteger {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.magnitude.hash(state);
        self.sign.hash(state);
    }
}

impl Display for BigInteger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}
/// Implements the addition operator for references to `BigInteger`.
impl<'a, 'b> core::ops::Add<&'b BigInteger> for &'a BigInteger {
    type Output = BigInteger;

    fn add(self, rhs: &'b BigInteger) -> BigInteger {
        self.add(rhs)
    }
}
//

pub static ZERO: LazyLock<BigInteger> = LazyLock::new(|| BigInteger::with_u32(0));
pub static ONE: LazyLock<BigInteger> = LazyLock::new(|| BigInteger::with_u32(1));
pub static TWO: LazyLock<BigInteger> = LazyLock::new(|| BigInteger::with_u32(2));
pub static THREE: LazyLock<BigInteger> = LazyLock::new(|| BigInteger::with_u32(3));
pub static FOUR: LazyLock<BigInteger> = LazyLock::new(|| BigInteger::with_u32(4));
pub static FIVE: LazyLock<BigInteger> = LazyLock::new(|| BigInteger::with_u32(5));
pub static SIX: LazyLock<BigInteger> = LazyLock::new(|| BigInteger::with_u32(6));
pub static SEVEN: LazyLock<BigInteger> = LazyLock::new(|| BigInteger::with_u32(7));
pub static EIGHT: LazyLock<BigInteger> = LazyLock::new(|| BigInteger::with_u32(8));
pub static NINE: LazyLock<BigInteger> = LazyLock::new(|| BigInteger::with_u32(9));
pub static TEN: LazyLock<BigInteger> = LazyLock::new(|| BigInteger::with_u32(10));
pub static SMALL_CONSTANTS: LazyLock<[BigInteger; 17]> = LazyLock::new(|| {
    let result: [BigInteger; 17] = [
        BigInteger::with_u32(0),
        BigInteger::with_u32(1),
        BigInteger::with_u32(2),
        BigInteger::with_u32(3),
        BigInteger::with_u32(4),
        BigInteger::with_u32(5),
        BigInteger::with_u32(6),
        BigInteger::with_u32(7),
        BigInteger::with_u32(8),
        BigInteger::with_u32(9),
        BigInteger::with_u32(10),
        BigInteger::with_u32(11),
        BigInteger::with_u32(12),
        BigInteger::with_u32(13),
        BigInteger::with_u32(14),
        BigInteger::with_u32(15),
        BigInteger::with_u32(16),
    ];
    result
});
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
//

fn calc_bit_length(sign: i8, magnitude: &[u32]) -> usize {
    if magnitude.is_empty() {
        return 0;
    }

    let mut index = 0usize;
    for i in 0..magnitude.len() {
        if magnitude[i] != 0 {
            break;
        }
        index += 1;
    }

    // a bit of length for everything after the first int
    let mut bit_length = 32 * ((magnitude.len() - index) - 1);

    // and determine the length of first int a bit
    let first_mag = magnitude[index];
    bit_length += bit_len(first_mag);

    // Check for negative powers of two
    if sign < 0 && ((first_mag & (-(first_mag as i64)) as u32) == first_mag) {
        loop {
            if {
                index += 1;
                index
            } >= magnitude.len()
            {
                bit_length -= 1;
                break;
            }
            if magnitude[index] == 0 {
                // nothing
            } else {
                break;
            }
        }
    }
    bit_length
}
fn bit_len(v: u32) -> usize {
    (size_of::<u32>() * 8) - (v.leading_zeros() as usize)
}
/// strip leading zeros
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
    if find { &buffer[pos..] } else { &buffer[pos..0] }
}
/// returns x = x - y - we assume x is >= y
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
            & I_MASK)
            - (y[{
                iv -= 1;
                iv as usize
            }] as i64
                & I_MASK)
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
fn do_sub_big_lil(big: &[u32], lil: &[u32]) -> Vec<u32> {
    let mut res = big.to_vec();
    subtract(&mut res, lil);
    res
}
/// return a = a + b - b preserved.
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
    0
}
fn init_be(buffer: &[u8]) -> (i8, Vec<u32>) {
    if (buffer[0] as i8) >= 0 {
        let magnitude = make_magnitude_be(buffer);
        let sign = if magnitude.is_empty() { 0 } else { 1 };
        (sign, magnitude)
    } else {
        let magnitude = make_magnitude_be_negative(buffer);
        let sign = -1;
        (sign, magnitude)
    }
}
fn init_le(buffer: &[u8]) -> (i8, Vec<u32>) {
    if (buffer[buffer.len() - 1] as i8) >= 0 {
        let magnitude = make_magnitude_le(buffer);
        let sign = if magnitude.is_empty() { 0 } else { 1 };
        (sign, magnitude)
    } else {
        let magnitude = make_magnitude_le_negative(buffer);
        let sign = -1;
        (sign, magnitude)
    }
}
/// make magnitude
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
        return vec![];
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
        return vec![];
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
        magnitude[i] = u32::from_le_bytes(sub_slice[pos..(pos + C_BYTES_PRE_INT)].try_into().unwrap());
    }
    magnitude
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
    inverse[0] += 1;
    make_magnitude_le(&inverse)
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
    if find { &buffer[..pos] } else { &buffer[pos..pos] }
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
/// return w with w = x * x - w is assumed to have enough space.
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
                & U_I_MASK)
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
/// return x with x = y * z - x is assumed to have enough space.
fn multiply(x: &mut [u32], y: &[u32], z: &[u32]) {
    let mut i = z.len();
    if i < 1 {
        return;
    }

    let mut x_base = x.len() as isize - y.len() as isize;

    loop {
        i -= 1;
        let a = z[i] as i64 & I_MASK;
        let mut val = 0i64;

        if a != 0 {
            for j in (0..y.len()).rev() {
                val += a.wrapping_mul(y[j] as i64 & I_MASK) + (x[(x_base + j as isize) as usize] as i64 & I_MASK);
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
/// return z = x / y - done in place (z value preserved, x contains the remainder)
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

    let mut xy_cmp = compare_no_leading_zeros(&x[x_start..], &y[y_start..]);
    let mut count: Vec<u32>;

    if xy_cmp > 0 {
        let y_bit_length = calc_bit_length(1, &y[y_start..]);
        let mut x_bit_length = calc_bit_length(1, &x[x_start..]);
        let mut shift = x_bit_length as isize - y_bit_length as isize;

        let mut i_count: Vec<u32>;
        let mut i_count_start = 0;

        let mut c: Vec<u32>;
        let mut c_start = 0;
        let mut c_bit_length = y_bit_length;
        if shift > 0 {
            i_count = vec![0u32; (shift as usize >> 5) + 1];
            i_count[0] = 1u32 << (shift as u32 % 32);

            c = shift_left(y, shift as usize);
            c_bit_length += shift as usize;
        } else {
            i_count = vec![1u32];
            let len = y.len() - y_start;

            c = vec![0u32; len];
            c.copy_from_slice(&y[y_start..]);
        }
        count = vec![0u32; i_count.len()];
        loop {
            if c_bit_length < x_bit_length || compare_no_leading_zeros(&x[x_start..], &c[c_start..]) >= 0 {
                subtract(&mut x[x_start..], &c[c_start..]);
                add_magnitudes(&mut count, &i_count);
                while x[x_start] == 0 {
                    x_start += 1;
                    if x_start == x.len() {
                        return count;
                    }
                }
                x_bit_length = 32 * (x.len() - x_start - 1) + bit_len(x[x_start]);
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
                let first_x = x[x_start];
                if first_c > first_x {
                    shift += 1;
                }
            }
            if shift < 2 {
                shift_right_one_in_place(&mut c[c_start..]);
                c_bit_length -= 1;
                shift_right_one_in_place(&mut i_count[i_count_start..])
            } else {
                shift_right_in_place(&mut c[c_start..], shift);
                c_bit_length -= shift as usize;
                shift_right_in_place(&mut i_count[i_count_start..], shift);
            }

            while c[c_start] == 0 {
                c_start += 1;
            }

            while i_count[i_count_start] == 0 {
                i_count_start += 1;
            }
        }
    } else {
        count = vec![0u32];
    }
    if xy_cmp == 0 {
        add_magnitudes(count.as_mut_slice(), &(*ONE).magnitude);
        for i in &mut x[x_start..] {
            *i = 0;
        }
    }
    count
}
/// do a right shift by one - this does it in place.
fn shift_right_one_in_place(mag: &mut [u32]) {
    let mut i = mag.len();
    let mut m = mag[i - 1];

    while {
        i -= 1;
        i
    } > 0
    {
        let next = mag[i - 1];
        mag[i] = (m >> 1) | (next << 31);
        m = next;
    }

    mag[0] = mag[0] >> 1;
}

/// do a right shift - this does it in place.
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
            if c_bit_length < x_bit_length || compare_no_leading_zeros(&x[x_start..], &c[c_start..]) >= 0 {
                subtract(&mut x[x_start..], &c[c_start..]);
                while x[x_start] == 0 {
                    x_start += 1;
                    if x_start == x.len() {
                        return;
                    }
                }

                x_bit_length = 32 * (x.len() - x_start - 1) + bit_len(x[x_start]);
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
fn get_bytes_length(n_bits: usize) -> usize {
    (n_bits + BITS_PER_BYTE - 1) / BITS_PER_BYTE
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

            carry += (prod2 & U_I_MASK) + ((prod1 as u32) << 1) as u64;
            a[(j + 2) as usize] = carry as u32;
            carry = (carry >> 32) + (prod1 >> 31) + (prod2 >> 32);
        }

        a[1] = carry as u32;
        a_max = (carry >> 32) as u32;
    }

    for i in (0..=n - 2).rev() {
        let a0 = a[n as usize];
        let t = a0.wrapping_mul(m_dash) as u64;
        let mut carry = t.wrapping_mul(m[(n - 1) as usize] as u64).wrapping_add(a0 as u64);
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

            carry += (prod1 & U_I_MASK) + (prod2 as u32) as u64 + a[(i + 1) as usize] as u64;
            a[(i + 2) as usize] = carry as u32;
            carry = (carry >> 32) + (prod1 >> 32) + (prod2 >> 32);
        }

        for j in (0..=(i - 1)).rev() {
            let prod1 = xi * x[j as usize] as u64;
            let prod2 = t * m[j as usize] as u64;

            carry += (prod2 & U_I_MASK) + ((prod1 as u32) << 1) as u64 + a[(j + 1) as usize] as u64;
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
fn get_window_list(mag: &[u32], extra_bits: usize) -> Vec<u32> {
    let mut v = mag[0];
    debug_assert!(v != 0);
    let leading_bits = bit_len(v);
    let total_bits = ((mag.len() - 1) << 5) + leading_bits;
    let result_size = (total_bits + extra_bits) / (1 + extra_bits) + 1;
    let mut result = vec![0u32; result_size];
    let mut result_pos = 0;
    let mut bit_pos = 33 - leading_bits;
    v = v.wrapping_shl(bit_pos as u32);

    let mut mul_t = 1;
    let mul_t_limit = 1 << extra_bits;
    let mut zeros = 0;

    let mut i = 0;
    loop {
        while bit_pos < 32 {
            bit_pos += 1;

            if mul_t < mul_t_limit {
                mul_t = (mul_t << 1) | (v >> 31);
            } else if (v as i32) < 0 {
                result[result_pos] = create_window_entry(mul_t, zeros);
                result_pos += 1;
                mul_t = 1;
                zeros = 0;
            } else {
                zeros += 1;
            }

            v <<= 1;
        }

        i += 1;
        if i == mag.len() {
            result[result_pos] = create_window_entry(mul_t, zeros);
            result_pos += 1;
            break;
        }

        v = mag[i];
        bit_pos = 0;
    }

    result[result_pos] = u32::MAX;
    result
}
fn create_window_entry(mut mul_t: u32, mut zeros: u32) -> u32 {
    debug_assert!(mul_t > 0);
    let tz = mul_t.trailing_zeros();
    mul_t >>= tz;
    zeros += tz;
    mul_t | (zeros << 8)
}
/// Montgomery multiplication: a = x * y * R^(-1) mod m
/// Based algorithm 14.36 of Handbook of Applied Cryptography.
/// - m, x, y should have length n
/// - a should have length (n + 1)
/// - b = 2^32, R = b^n
/// The result is put in x
/// NOTE: the indices of x, y, m, a different in HAC and in Java
fn multiply_monty(a: &mut [u32], x: &mut [u32], y: &[u32], m: &[u32], m_dash: u32, small_monty_modulus: bool) {
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
            let prod1 = xi * y[j] as u64;
            prod2 = t.wrapping_mul(m[j] as u64);

            carry += (prod1 & U_I_MASK) + (prod2 as u32) as u64;
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
        let mut carry = (prod1 & U_I_MASK) + a0 as u64;
        let t = (carry as u32).wrapping_mul(m_dash) as u64;

        let mut prod2 = t.wrapping_mul(m[n - 1] as u64);
        carry += (prod2 as u32) as u64;
        debug_assert!(carry as u32 == 0);
        carry = (carry >> 32) + (prod1 >> 32) + (prod2 >> 32);

        for j in (0..=(n - 2)).rev() {
            prod1 = xi * y[j] as u64;
            prod2 = t.wrapping_mul(m[j] as u64);

            carry += (prod1 & U_I_MASK) + (prod2 as u32) as u64 + a[j + 1] as u64;
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
    // NOTE: Not a general purpose reduction (which would allow x up to twice the bit length of m)
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
fn append_zero_extended_string(sb: &mut String, s: &str, min_length: usize) {
    let mut len = s.len();
    while len < min_length {
        sb.push('0');
        len += 1;
    }
    sb.push_str(s);
}
fn to_string_with_moduli(sb: &mut String, radix: u32, moduli: &[BigInteger], mut scale: usize, pos: &BigInteger) {
    if pos.bit_length() < 64 {
        let s = match radix {
            2 => format!("{:b}", pos.as_i64()),
            8 => format!("{:o}", pos.as_i64()),
            10 => format!("{}", pos.as_i64()),
            16 => format!("{:X}", pos.as_i64()),
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
    let (qr1, qr2) = pos.divide_and_remainder(&moduli[scale]).unwrap();
    to_string_with_moduli(sb, radix, moduli, scale, &qr1);
    to_string_with_moduli(sb, radix, moduli, scale, &qr2);
}
/// Calculate the numbers u1, u2, and u3 such that:
///
/// U1 * a + u2 * b = u3
///
/// Where u3 is the greatest common divider of a and b. a and b using the extended Euclid algorithm
/// (refer p. 323 of The Art of Computer Programming vol. 2, 2nd ed.).
/// This also seems to have the side effect of calculating some form of multiplicative inverse.
fn ext_euclid(a: &BigInteger, b: &BigInteger) -> (BigInteger, BigInteger) {
    let mut u1 = (*ONE).clone();
    let mut v1 = (*ZERO).clone();
    let mut u3 = a.clone();
    let mut v3 = b.clone();

    if v3.sign() > 0 {
        loop {
            let q = u3.divide_and_remainder(&v3).unwrap();
            u3 = v3;
            v3 = q.1;

            let old_u1 = u1;
            u1 = v1.clone();
            if v3.sign() <= 0 {
                break;
            }
            v1 = old_u1.subtract(&v1.multiply(&q.0));
        }
    }
    (u3, u1)
}
fn create_value_of_u32(value: u32) -> BigInteger {
    if value == 0 {
        return (*ZERO).clone();
    }
    BigInteger::with_check_magnitude(1, vec![value], false)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_bug_81857() {
        let b = BigInteger::with_string("18446744073709551616").expect("error");
        //let exp = (*TWO).clone();
        let mod_ = BigInteger::with_string("48112959837082048697").expect("error");
        let expected = BigInteger::with_string("4970597831480284165").expect("error");

        let manual = b.multiply(&b).modulus(&mod_).unwrap();
        assert_eq!(expected, manual, "b * b % mod");
    }

    static MINUS_ONE: LazyLock<BigInteger> = LazyLock::new(|| BigInteger::with_i32(-1));
    static MINUS_TWO: LazyLock<BigInteger> = LazyLock::new(|| BigInteger::with_i32(-2));
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
        assert_eq!(0, (*ZERO).bit_count());
        assert_eq!(1, (*ONE).bit_count());
        assert_eq!(0, (*MINUS_ONE).bit_count());
        assert_eq!(1, (*TWO).bit_count());
        assert_eq!(1, (*MINUS_TWO).bit_count());
        for i in 0..100 {
            let pow2 = (*ONE).shift_left(i);
            assert_eq!(1, pow2.bit_count());
            assert_eq!(i as usize, pow2.negate().bit_count(), "{}", i);
        }

        let mut random = rand::rng();

        for _ in 0..10 {
            let test = BigInteger::with_random_certainty(128, 0, &mut random).unwrap();
            let mut bit_count = 0usize;

            //println!("bit length: {}, bit count: {}", *test.get_bit_length(), *test.get_bit_count());

            for bit in 0..test.bit_length() {
                if test.test_bit(bit) {
                    bit_count += 1;
                }
            }

            assert_eq!(bit_count, test.bit_count());
        }
    }

    #[test]
    fn test_bit_length() {
        assert_eq!(0, (*ZERO).bit_length());
        assert_eq!(1, (*ONE).bit_length());
        assert_eq!(0, (*MINUS_ONE).bit_length());
        assert_eq!(2, (*TWO).bit_length());
        assert_eq!(1, (*MINUS_TWO).bit_length());

        let mut random = rand::rng();

        for i in 0..100 {
            let bit = i + rand::random_range(0..64);
            //println!("{}", bit);
            let odd = BigInteger::with_random(bit, &mut random).set_bit(bit + 1).set_bit(0);
            let pow2 = (*ONE).shift_left(bit as isize);

            assert_eq!(bit + 2, odd.bit_length(), "t1");
            assert_eq!(bit + 2, odd.negate().bit_length(), "t2");
            assert_eq!(bit + 1, pow2.bit_length(), "t3");
            assert_eq!(bit, pow2.negate().bit_length(), "t4");
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

        let mut random = rand::rng();
        for _ in 0..10 {
            let n = BigInteger::with_random(128, &mut random);

            for _ in 0..10 {
                let pos = rand::random_range(0..128);
                let m = n.clear_bit(pos);
                assert_ne!(m.shift_right(pos as isize).remainder(&(*TWO)).unwrap(), *ONE);
            }
        }

        for i in 0..100 {
            let pow2 = (*ONE).shift_left(i);
            let minus_pow2 = pow2.negate();

            assert_eq!(*ZERO, pow2.clear_bit(i as usize));

            let right = minus_pow2.clear_bit(i as usize);
            assert_eq!(minus_pow2.shift_left(1), right, "i = {}, minus_pow2 = {:?}", i, minus_pow2);

            let big_i = BigInteger::with_i32(i as i32);
            let neg_i = big_i.negate();

            for j in 0..10 {
                assert_eq!(big_i.and_not(&(*ONE).shift_left(j)), big_i.clear_bit(j as usize), "i = {}, j = {}", i, j);
                assert_eq!(neg_i.and_not(&(*ONE).shift_left(j)), neg_i.clear_bit(j as usize), "i = {}, j = {}", i, j);
            }
        }
    }

    #[test]
    fn test_compare_to() {
        assert_eq!(Some(Ordering::Equal), (*MINUS_TWO).partial_cmp(&(*MINUS_TWO)));
        assert_eq!(Some(Ordering::Less), (*MINUS_TWO).partial_cmp(&(*MINUS_ONE)));
        assert_eq!(Some(Ordering::Less), (*MINUS_TWO).partial_cmp(&(*ZERO)));
        assert_eq!(Some(Ordering::Less), (*MINUS_TWO).partial_cmp(&(*ONE)));
        assert_eq!(Some(Ordering::Less), (*MINUS_TWO).partial_cmp(&(*TWO)));

        assert_eq!(Some(Ordering::Greater), (*MINUS_ONE).partial_cmp(&(*MINUS_TWO)));
        assert_eq!(Some(Ordering::Equal), (*MINUS_ONE).partial_cmp(&(*MINUS_ONE)));
        assert_eq!(Some(Ordering::Less), (*MINUS_ONE).partial_cmp(&(*ZERO)));
        assert_eq!(Some(Ordering::Less), (*MINUS_ONE).partial_cmp(&(*ONE)));
        assert_eq!(Some(Ordering::Less), (*MINUS_ONE).partial_cmp(&(*TWO)));

        assert_eq!(Some(Ordering::Greater), (*ZERO).partial_cmp(&(*MINUS_TWO)));
        assert_eq!(Some(Ordering::Greater), (*ZERO).partial_cmp(&(*MINUS_ONE)));
        assert_eq!(Some(Ordering::Equal), (*ZERO).partial_cmp(&(*ZERO)));
        assert_eq!(Some(Ordering::Less), (*ZERO).partial_cmp(&(*ONE)));
        assert_eq!(Some(Ordering::Less), (*ZERO).partial_cmp(&(*TWO)));

        assert_eq!(Some(Ordering::Greater), (*ONE).partial_cmp(&(*MINUS_TWO)));
        assert_eq!(Some(Ordering::Greater), (*ONE).partial_cmp(&(*MINUS_ONE)));
        assert_eq!(Some(Ordering::Greater), (*ONE).partial_cmp(&(*ZERO)));
        assert_eq!(Some(Ordering::Equal), (*ONE).partial_cmp(&(*ONE)));
        assert_eq!(Some(Ordering::Less), (*ONE).partial_cmp(&(*TWO)));

        assert_eq!(Some(Ordering::Greater), (*TWO).partial_cmp(&(*MINUS_TWO)));
        assert_eq!(Some(Ordering::Greater), (*TWO).partial_cmp(&(*MINUS_ONE)));
        assert_eq!(Some(Ordering::Greater), (*TWO).partial_cmp(&(*ZERO)));
        assert_eq!(Some(Ordering::Greater), (*TWO).partial_cmp(&(*ONE)));
        assert_eq!(Some(Ordering::Equal), (*TWO).partial_cmp(&(*TWO)));
    }

    #[test]
    fn test_constructors() {
        assert_eq!(&(*ZERO), &BigInteger::with_buffer(&[0u8]));
        assert_eq!(&(*ZERO), &BigInteger::with_buffer(&[0u8, 0u8]));

        let mut random = rand::rng();
        for i in 0..10 {
            let m = BigInteger::with_random_certainty(i + 3, 0, &mut random).unwrap().test_bit(0);
            assert!(m, "i = {}", i);
        }
    }

    #[test]
    fn test_divide_01() {
        for i in -5..=5 {
            let m = BigInteger::with_i32(i);
            assert!(m.divide(&(*ZERO)).is_err());
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
                big_product.divide(&BigInteger::with_i32(divisor)).unwrap(),
                "divisor = {}",
                divisor
            );
            assert_eq!(expected.negate(), big_product.negate().divide(&BigInteger::with_i32(divisor)).unwrap());
            assert_eq!(expected.negate(), big_product.divide(&BigInteger::with_i32(divisor).negate()).unwrap());
            assert_eq!(expected, big_product.negate().divide(&BigInteger::with_i32(divisor).negate()).unwrap());

            let expected = BigInteger::with_i32((product + 1) / divisor);

            assert_eq!(expected, big_product_plus.divide(&BigInteger::with_i32(divisor)).unwrap());
            assert_eq!(
                expected.negate(),
                big_product_plus.negate().divide(&BigInteger::with_i32(divisor)).unwrap()
            );
            assert_eq!(
                expected.negate(),
                big_product_plus.divide(&BigInteger::with_i32(divisor).negate()).unwrap()
            );
            assert_eq!(
                expected,
                big_product_plus.negate().divide(&BigInteger::with_i32(divisor).negate()).unwrap()
            );
        }
    }

    #[test]
    fn test_divide_03() {
        let mut random = rand::rng();
        for req in 0..10 {
            let a = BigInteger::with_random_certainty(100 - req, 0, &mut random).unwrap();
            let b = BigInteger::with_random_certainty(100 + req, 0, &mut random).unwrap();
            let c = BigInteger::with_random_certainty(10 + req, 0, &mut random).unwrap();
            let d = a.multiply(&b).add(&c);
            let e = d.divide(&a).unwrap();

            assert_eq!(b, e);
        }

        // Special tests for power of two since uses a different code path internally
        for _ in 0..100 {
            let shift = rand::random_range(0..64) as isize;
            let a = (*ONE).shift_left(shift);
            let b = BigInteger::with_random(64 + rand::random_range(0..64), &mut random);
            let b_shift = b.shift_right(shift);

            assert_eq!(b_shift, b.divide(&a).unwrap(), "shift = {}, b = {:?}", shift, b.to_string_radix(16));
            assert_eq!(
                b_shift.negate(),
                b.divide(&a.negate()).unwrap(),
                "shift = {}, b = {:?}",
                shift,
                b.to_string_radix(16)
            );
            assert_eq!(
                b_shift.negate(),
                b.negate().divide(&a).unwrap(),
                "shift = {}, b = {:?}",
                shift,
                b.to_string_radix(16)
            );
            assert_eq!(
                b_shift,
                b.negate().divide(&a.negate()).unwrap(),
                "shift = {}, b = {:?}",
                shift,
                b.to_string_radix(16)
            );
        }
    }

    #[test]
    fn test_divide_04() {
        // Regression
        let shift = 63;
        let a = (*ONE).shift_left(shift);
        let b = BigInteger::with_i64(0x2504b470dc188499);
        let b_shift = b.shift_right(shift);

        assert_eq!(b_shift, b.divide(&a).unwrap(), "shift = {}, b = {:?}", shift, b.to_string_radix(16));
        assert_eq!(
            b_shift.negate(),
            b.divide(&a.negate()).unwrap(),
            "shift = {}, b = {:?}",
            shift,
            b.to_string_radix(16)
        );
        assert_eq!(
            b_shift.negate(),
            b.negate().divide(&a).unwrap(),
            "shift = {}, b = {:?}",
            shift,
            b.to_string_radix(16)
        );
        assert_eq!(
            b_shift,
            b.negate().divide(&a.negate()).unwrap(),
            "shift = {}, b = {:?}",
            shift,
            b.to_string_radix(16)
        );
    }

    #[test]
    fn test_divide_and_remainder() {
        let mut random = rand::rng();
        let n = BigInteger::with_random(48, &mut random);
        let mut qr = n.divide_and_remainder(&n).unwrap();
        assert_eq!(*ONE, qr.0);
        assert_eq!(*ZERO, qr.1);

        qr = n.divide_and_remainder(&(*ONE)).unwrap();
        assert_eq!(&n, &qr.0);
        assert_eq!(&(*ZERO), &qr.1);

        for rep in 0..10 {
            let a = BigInteger::with_random_certainty(100 - rep, 0, &mut random).unwrap();
            let b = BigInteger::with_random_certainty(100 + rep, 0, &mut random).unwrap();
            let c = BigInteger::with_random_certainty(10 + rep, 0, &mut random).unwrap();
            let d = a.multiply(&b).add(&c);
            let es = d.divide_and_remainder(&a).unwrap();

            assert_eq!(&b, &es.0);
            assert_eq!(&c, &es.1);
        }

        // Special tests for power of two since uses a different code path internally
        for _ in 0..100 {
            let shift = rand::random_range(0..64) as isize;
            let a = (*ONE).shift_left(shift);
            let b = BigInteger::with_random(64 + rand::random_range(0..64), &mut random);
            let b_shift = b.shift_right(shift);
            let b_mod = b.and(&a.subtract(&(*ONE)));

            qr = b.divide_and_remainder(&a).unwrap();
            assert_eq!(b_shift, qr.0, "shift = {}, b = {:?}", shift, b.to_string_radix(16));
            assert_eq!(b_mod, qr.1, "shift = {}, b = {:?}", shift, b.to_string_radix(16));

            qr = b.divide_and_remainder(&a.negate()).unwrap();
            assert_eq!(b_shift.negate(), qr.0, "shift = {}, b = {:?}", shift, b.to_string_radix(16));
            assert_eq!(b_mod, qr.1, "shift = {}, b = {:?}", shift, b.to_string_radix(16));

            qr = b.negate().divide_and_remainder(&a).unwrap();
            assert_eq!(b_shift.negate(), qr.0, "shift = {}, b = {:?}", shift, b.to_string_radix(16));
            assert_eq!(b_mod.negate(), qr.1, "shift = {}, b = {:?}", shift, b.to_string_radix(16));

            qr = b.negate().divide_and_remainder(&a.negate()).unwrap();
            assert_eq!(b_shift, qr.0, "shift = {}, b = {:?}", shift, b.to_string_radix(16));
            assert_eq!(b_mod.negate(), qr.1, "shift = {}, b = {:?}", shift, b.to_string_radix(16));
        }
    }

    #[test]
    fn test_flip_bit() {
        let mut random = rand::rng();
        for _ in 0..10 {
            let a = BigInteger::with_random_certainty(128, 0, &mut random).unwrap();
            let b = a.clone();

            for _ in 0..100 {
                // Note: Intentionally greater than initial size
                let pos = rand::random_range(0..256);

                let a = a.flip_bit(pos);
                let b = if b.test_bit(pos) { b.clear_bit(pos) } else { b.set_bit(pos) };

                assert_eq!(a, b);
            }
        }

        for i in 0..100 {
            let pow2 = (*ONE).shift_left(i);
            let minus_pow2 = pow2.negate();

            assert_eq!(*ZERO, pow2.flip_bit(i as usize));
            assert_eq!(minus_pow2.shift_left(1), minus_pow2.flip_bit(i as usize), "i = {}", i);

            let big_i = BigInteger::with_i32(i as i32);
            let neg_i = big_i.negate();

            for j in 0..10 {
                assert_eq!(big_i.xor(&(*ONE).shift_left(j)), big_i.flip_bit(j as usize), "i = {}, j = {}", i, j);
                assert_eq!(neg_i.xor(&(*ONE).shift_left(j)), neg_i.flip_bit(j as usize), "i = {}, j = {}", i, j);
            }
        }
    }

    #[test]
    fn test_gcd() {
        let mut random = rand::rng();
        for _ in 0..10 {
            let fac = BigInteger::with_random(32, &mut random).add(&(*TWO));
            let p1 = BigInteger::with_probable_prime(63, &mut random).unwrap();
            let p2 = BigInteger::with_probable_prime(64, &mut random).unwrap();
            let gcd = fac.multiply(&p1).gcd(&fac.multiply(&p2)).unwrap();

            assert_eq!(fac, gcd);
        }
    }

    #[test]
    fn tst_get_lowest_set_bit() {
        let mut random = rand::rng();
        for i in 1..=100 {
            let test = BigInteger::with_random_certainty(i + 1, 0, &mut random).unwrap();
            let bit1 = test.get_lowest_set_bit();
            assert_eq!(test, test.shift_right(bit1 as isize).shift_left(bit1 as isize));
            let bit2 = test.shift_left(i as isize + 1).get_lowest_set_bit();
            assert_eq!(i as i32 + 1, bit2 - bit1);
            let bit3 = test.shift_left(3 * i as isize).get_lowest_set_bit();
            assert_eq!(3 * i as i32, bit3 - bit1);
        }
    }

    #[test]
    fn test_i32_value() {
        let tests = [i32::MIN, -1234, -10, -1, 0, !0, 1, 10, 5678, i32::MAX];
        for i in tests.iter() {
            let a = BigInteger::with_i32(*i);
            assert_eq!(*i, a.as_i32(), "i = {}", i);
        }
    }

    const FIRST_PRIMES: [i32; 10] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29];
    const NON_PRIMES: [i32; 10] = [0, 1, 4, 10, 20, 21, 22, 25, 26, 27];
    const MERSENNE_PRIME_EXPONENTS: [i32; 10] = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89];
    const NON_PRIME_EXPONENTS: [i32; 10] = [1, 4, 6, 9, 11, 15, 23, 29, 37, 41];
    #[test]
    fn test_is_probable_prime() {
        assert!(!&(*ZERO).is_probable_prime(100).unwrap());
        assert!(&(*ZERO).is_probable_prime(0).unwrap());
        assert!(&(*ZERO).is_probable_prime(-10).unwrap());
        assert!(!&(*MINUS_ONE).is_probable_prime(100).unwrap());
        assert!(&(*MINUS_TWO).is_probable_prime(100).unwrap());
        assert!(BigInteger::with_i32(-17).is_probable_prime(100).unwrap());
        assert!(BigInteger::with_i32(67).is_probable_prime(100).unwrap());
        assert!(BigInteger::with_i32(773).is_probable_prime(100).unwrap());

        for p in FIRST_PRIMES {
            assert!(BigInteger::with_i32(p).is_probable_prime(100).unwrap());
            assert!(BigInteger::with_i32(-p).is_probable_prime(100).unwrap());
        }

        for c in NON_PRIMES {
            assert!(!BigInteger::with_i32(c).is_probable_prime(100).unwrap());
            assert!(!BigInteger::with_i32(-c).is_probable_prime(100).unwrap());
        }

        for e in MERSENNE_PRIME_EXPONENTS {
            assert!(
                &(*TWO).pow(e as u32).unwrap().subtract(&(*ONE)).is_probable_prime(100).unwrap(),
                "e = {}",
                e
            );
            assert!(&(*TWO).pow(e as u32).unwrap().subtract(&(*ONE)).negate().is_probable_prime(100).unwrap());
        }

        for e in NON_PRIME_EXPONENTS {
            assert!(!(&(*TWO).pow(e as u32).unwrap().subtract(&(*ONE)).is_probable_prime(100).unwrap()));
            assert!(!(&(*TWO).pow(e as u32).unwrap().subtract(&(*ONE)).negate().is_probable_prime(100).unwrap()));
        }
    }

    #[test]
    fn test_long_value() {
        let tests = [i64::MIN, -1234, -10, -1, 0, !0, 1, 10, 5678, i64::MAX];
        for i in tests.iter() {
            let a = BigInteger::with_i64(*i);
            assert_eq!(*i, a.as_i64(), "i = {}", i);
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
        let mut random = rand::rng();
        for _ in 0..100 {
            let diff = rand::random_range(0..25);
            let a = BigInteger::with_random_certainty(100 - diff, 0, &mut random).unwrap();
            let b = BigInteger::with_random_certainty(100 + diff, 0, &mut random).unwrap();
            let c = BigInteger::with_random_certainty(10 + diff, 0, &mut random).unwrap();

            let d = a.multiply(&b).add(&c);
            let e = d.remainder(&a).unwrap();
            assert_eq!(c, e);

            let pow2 = (*ONE).shift_left(rand::random_range(0..128) as isize);
            assert_eq!(b.and(&pow2.subtract(&(*ONE))), b.modulus(&pow2).unwrap());
        }
    }

    #[test]
    fn test_mod_inverse() {
        let mut random = rand::rng();
        for _ in 0..10 {
            let p = BigInteger::with_probable_prime(64, &mut random).unwrap();
            let q = BigInteger::with_random(63, &mut random).add(&(*ONE));
            let inv = q.mod_inverse(&p).unwrap();
            let inv2 = inv.mod_inverse(&p).unwrap();

            assert_eq!(q, inv2);
            assert_eq!(*ONE, q.multiply(&inv).modulus(&p).unwrap());
        }

        // ModInverse a power of 2 for a range of powers
        for i in 1..=128 {
            let m = (*ONE).shift_left(i as isize);
            let d = BigInteger::with_random(i, &mut random).set_bit(0);
            let x = d.mod_inverse(&m).unwrap();
            let check = x.multiply(&d).modulus(&m).unwrap();

            assert_eq!(*ONE, check, "i = {}, x = {:?}, d = {:?}", i, x, d);
        }
    }

    #[test]
    fn test_mod_pow_01() {
        assert!((*TWO).mod_pow(&(*ONE), &(*ZERO)).is_err());
    }

    #[test]
    fn test_mod_pow_02() {
        assert_eq!(*ZERO, (*ZERO).mod_pow(&(*ZERO), &(*ONE)).unwrap());
        assert_eq!(*ONE, (*ZERO).mod_pow(&(*ZERO), &(*TWO)).unwrap());
        assert_eq!(*ZERO, (*TWO).mod_pow(&(*ONE), &(*ONE)).unwrap());
        assert_eq!(*ONE, (*TWO).mod_pow(&(*ZERO), &(*TWO)).unwrap());

        let mut random = rand::rng();
        for i in 0..100 {
            let m = BigInteger::with_probable_prime(10 + i, &mut random).unwrap();
            let x = BigInteger::with_random(m.bit_length() - 1, &mut random);
            assert_eq!(x, x.mod_pow(&m, &m).unwrap(), "i = {}, x = {:?}, m = {:?}", i, x, m);

            if x.sign() != 0 {
                assert_eq!(*ZERO, (*ZERO).mod_pow(&x, &m).unwrap());
                assert_eq!(*ONE, x.mod_pow(&m.subtract(&(*ONE)), &m).unwrap());
            }

            let y = BigInteger::with_random(m.bit_length() - 1, &mut random);
            let n = BigInteger::with_random(m.bit_length() - 1, &mut random);
            let n3 = n.mod_pow(&(*THREE), &m).unwrap();

            let res_x = n.mod_pow(&x, &m).unwrap();
            let res_y = n.mod_pow(&y, &m).unwrap();
            let res = res_x.multiply(&res_y).modulus(&m).unwrap();
            let res3 = res.mod_pow(&(*THREE), &m).unwrap();

            assert_eq!(res3, n3.mod_pow(&x.add(&y), &m).unwrap());

            let a = x.add(&(*ONE));
            let b = y.add(&(*ONE));

            assert_eq!(a.mod_pow(&b, &m).unwrap().mod_inverse(&m).unwrap(), a.mod_pow(&b.negate(), &m).unwrap());
        }
    }

    #[test]
    fn test_multiply() {
        let one = &(*ONE);
        assert_eq!(one, &one.negate().multiply(&one.negate()));

        let mut random = rand::rng();
        for _ in 0..100 {
            let a_len = 64 + rand::random_range(0..64);
            let b_len = 64 + rand::random_range(0..64);

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

        // Special tests for power of two since uses a different code path internally
        for _ in 0..100 {
            let shift = rand::random_range(0..64) as isize;
            let a = one.shift_left(shift);
            let b = BigInteger::with_random(64 + rand::random_range(0..64), &mut random);
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
        let mut random = rand::rng();
        let first_prime = BigInteger::with_probable_prime(32, &mut random).unwrap();
        let next_prime = first_prime.next_probable_prime().unwrap();

        assert!(first_prime.is_probable_prime(10).unwrap());
        assert!(next_prime.is_probable_prime(10).unwrap());

        let mut check = first_prime.add(&(*ONE));
        while check < next_prime {
            assert!(!check.is_probable_prime(10).unwrap());
            check = check.add(&(*ONE));
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
        assert_eq!(*ONE, (*ZERO).pow(0).unwrap());
        assert_eq!(*ZERO, (*ZERO).pow(123).unwrap());
        assert_eq!(*ONE, (*ONE).pow(0).unwrap());
        assert_eq!(*ONE, (*ONE).pow(123).unwrap());

        assert_eq!((*TWO).pow(147).unwrap(), (*ONE).shift_left(147));
        assert_eq!((*ONE).shift_left(7).pow(11).unwrap(), (*ONE).shift_left(77));

        let n = BigInteger::with_string("1234567890987654321").expect("error");
        let mut result = (*ONE).clone();
        for i in 0..10 {
            assert_eq!(result, n.pow(i).unwrap(), "i = {}", i);
            result = result.multiply(&n);
        }
    }

    #[test]
    fn test_remainder() {
        for rep in 0..10 {
            let a = BigInteger::with_random_certainty(100 - rep, 0, &mut rand::rng()).unwrap();
            let b = BigInteger::with_random_certainty(100 + rep, 0, &mut rand::rng()).unwrap();
            let c = BigInteger::with_random_certainty(10 + rep, 0, &mut rand::rng()).unwrap();
            let d = a.multiply(&b).add(&c);
            let e = d.remainder(&a).unwrap();
            let f = d.divide(&a).unwrap();

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

        let mut random = rand::rng();
        for _ in 0..10 {
            let n = BigInteger::with_random(128, &mut random);

            for _ in 0..10 {
                let pos = rand::random_range(0..128);
                let m = n.set_bit(pos);
                let test = m.shift_right(pos as isize).remainder(&(*TWO)).unwrap() == *ONE;
                assert!(test);
            }
        }
    }

    #[test]
    fn test_shift_left() {
        let mut random = rand::rng();
        for i in 0..100 {
            let shift = rand::random_range(0..128);

            let a = BigInteger::with_random(128 + i, &mut random);
            a.bit_count();

            let neg_a = a.negate();
            neg_a.bit_count();

            let b = a.shift_left(shift as isize);
            let c = neg_a.shift_left(shift as isize);

            assert_eq!(a.bit_count(), b.bit_count(), "1. i = {}, shift = {}", i, shift);
            assert_eq!(neg_a.bit_count() + shift, c.bit_count(), "2. i = {}, shift = {}", i, shift);
            assert_eq!(a.bit_length() + shift, b.bit_length());
            assert_eq!(neg_a.bit_length() + shift, c.bit_length());

            let mut j = 0usize;
            while j < shift {
                assert!(!b.test_bit(j));
                j += 1;
            }
            while j < (b.bit_length()) {
                assert_eq!(a.test_bit(j - shift), b.test_bit(j));
                j += 1;
            }
        }
    }

    #[test]
    fn test_shift_right() {
        let mut random = rand::rng();
        for i in 0..10 {
            let shift = rand::random_range(0..128);
            let a = BigInteger::with_random(256 + i, &mut random);
            let b = a.shift_right(shift as isize);

            assert_eq!(a.bit_length() - shift, b.bit_length());

            for j in 0..b.bit_length() {
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
            assert_eq!(b, a.sign());
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

        let mut random = rand::rng();
        for _ in 0..10 {
            let a = BigInteger::with_random_certainty(128, 0, &mut random).unwrap();
            let b = BigInteger::with_random_certainty(128, 0, &mut random).unwrap();
            let c = a.subtract(&b);
            let d = b.subtract(&a);
            assert_eq!(c.abs(), d.abs());
        }
    }

    #[test]
    fn test_test_bit() {
        let mut random = rand::rng();
        for _ in 0..10 {
            let n = BigInteger::with_random(128, &mut random);
            assert!(!n.test_bit(128));
            assert!(n.negate().test_bit(128));

            for _ in 0..10 {
                let pos = rand::random_range(0..128);
                let test = n.shift_right(pos as isize).remainder(&(*TWO)).unwrap() == *ONE;
                assert_eq!(test, n.test_bit(pos));
            }
        }
    }

    #[test]
    fn test_to_vec() {
        let z = &(*ZERO).to_vec();
        assert_eq!(z.len(), 1);
        assert_eq!(z[0], 0);

        let mut random = rand::rng();
        for i in 16..=48 {
            let x = BigInteger::with_random(i, &mut random).set_bit(i - 1);
            let b = x.to_vec();
            assert_eq!(b.len(), i / 8 + 1);
            let y = BigInteger::with_buffer(&b);
            assert_eq!(x, y);

            let x = x.negate();
            let b = x.to_vec();
            assert_eq!(b.len(), i / 8 + 1);
            let y = BigInteger::with_buffer(&b);
            assert_eq!(x, y);
        }
    }

    #[test]
    fn test_to_vec_unsigned() {
        let z = &(*ZERO).to_vec_unsigned();
        assert_eq!(z.len(), 0);

        let mut random = rand::rng();
        for i in 16..=48 {
            let mut x = BigInteger::with_random(i, &mut random).set_bit(i - 1);
            let mut b = x.to_vec_unsigned();
            assert_eq!(b.len(), (i + 7) / 8);
            let mut y = BigInteger::with_sign_buffer(1, &b).expect("Failed to create BigInteger from buffer");
            assert_eq!(x, y);

            x = x.negate();
            b = x.to_vec_unsigned();
            assert_eq!(b.len(), i / 8 + 1);
            y = BigInteger::with_buffer(&b);
            assert_eq!(x, y);
        }
    }

    #[test]
    fn test_to_string() {
        let s = "1234567890987654321";
        assert_eq!(s, &BigInteger::with_string(s).unwrap().to_string());
        assert_eq!(s, &BigInteger::with_string_radix(s, 10).unwrap().to_string_radix(10).unwrap());
        assert_eq!(s, &BigInteger::with_string_radix(s, 16).unwrap().to_string_radix(16).unwrap());

        let mut random = rand::rng();
        for i in 0..100 {
            let left = BigInteger::with_random(i, &mut random);
            {
                let right = BigInteger::with_string_radix(&left.to_string_radix(2).unwrap(), 2).unwrap();
                assert_eq!(
                    left,
                    right,
                    "radix = 2, i = {}, \n left: n2 = {}, n8 = {}, n10 = {}, n16 = {} \n right: n2 = {}, n8 = {}, n10 = {}, n16 = {}",
                    i,
                    left.to_string_radix(2).unwrap(),
                    left.to_string_radix(8).unwrap(),
                    left.to_string_radix(10).unwrap(),
                    left.to_string_radix(16).unwrap(),
                    right.to_string_radix(2).unwrap(),
                    right.to_string_radix(8).unwrap(),
                    right.to_string_radix(10).unwrap(),
                    right.to_string_radix(16).unwrap()
                );
            }
            {
                let right = BigInteger::with_string_radix(&left.to_string_radix(8).unwrap(), 8).unwrap();
                assert_eq!(
                    left,
                    right,
                    "radix = 8, i = {}, \n left: n2 = {}, n8 = {}, n10 = {}, n16 = {} \n right: n2 = {}, n8 = {}, n10 = {}, n16 = {}",
                    i,
                    left.to_string_radix(2).unwrap(),
                    left.to_string_radix(8).unwrap(),
                    left.to_string_radix(10).unwrap(),
                    left.to_string_radix(16).unwrap(),
                    right.to_string_radix(2).unwrap(),
                    right.to_string_radix(8).unwrap(),
                    right.to_string_radix(10).unwrap(),
                    right.to_string_radix(16).unwrap()
                );
            }
            {
                let right = BigInteger::with_string_radix(&left.to_string_radix(10).unwrap(), 10).unwrap();
                assert_eq!(
                    left,
                    right,
                    "radix = 10, i = {}, \n left: n2 = {}, n8 = {}, n10 = {}, n16 = {} \n right: n2 = {}, n8 = {}, n10 = {}, n16 = {}",
                    i,
                    left.to_string_radix(2).unwrap(),
                    left.to_string_radix(8).unwrap(),
                    left.to_string_radix(10).unwrap(),
                    left.to_string_radix(16).unwrap(),
                    right.to_string_radix(2).unwrap(),
                    right.to_string_radix(8).unwrap(),
                    right.to_string_radix(10).unwrap(),
                    right.to_string_radix(16).unwrap()
                );
            }
            let right = BigInteger::with_string_radix(&left.to_string_radix(16).unwrap(), 16).unwrap();
            assert_eq!(
                left,
                right,
                "radix = 2, i = {}, \n left: n2 = {}, n8 = {}, n10 = {}, n16 = {} \n right: n2 = {}, n8 = {}, n10 = {}, n16 = {}",
                i,
                left.to_string_radix(2).unwrap(),
                left.to_string_radix(8).unwrap(),
                left.to_string_radix(10).unwrap(),
                left.to_string_radix(16).unwrap(),
                right.to_string_radix(2).unwrap(),
                right.to_string_radix(8).unwrap(),
                right.to_string_radix(10).unwrap(),
                right.to_string_radix(16).unwrap()
            );
        }

        // Radix version
        let radices = [2, 8, 10, 16];
        let trials = 256;

        let mut tests = vec![(*ZERO).clone(); trials];
        for i in 0..trials {
            let len = rand::random_range(0..(i + 1));
            tests[i] = BigInteger::with_random(len, &mut random);
        }

        for radix in radices {
            for i in 0..trials {
                let n1 = &tests[i];
                let str = n1.to_string_radix(radix).unwrap();
                let n2 = BigInteger::with_string_radix(&str, radix).unwrap();
                assert_eq!(n1, &n2);
            }
        }
    }

    #[test]
    fn test_value_of() {
        assert_eq!(-1, BigInteger::with_i32(-1).sign());
        assert_eq!(0, BigInteger::with_i32(0).sign());
        assert_eq!(1, BigInteger::with_i32(1).sign());

        for i in -5..5 {
            let a = BigInteger::with_i32(i);
            assert_eq!(i, a.as_i32());
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
}
