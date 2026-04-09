//! Utility functions for 64-bit integer bit manipulation.

/// Number of bits in a `u64`.
pub const NUM_BITS: u32 = 64;
/// Number of bytes in a `u64`.
pub const NUM_BYTES: u32 = 8;

/// Returns the value of the highest one-bit in `i`.
///
/// # Examples
///
/// ```
/// use bc_rust::util::longs::highest_one_bit;
/// assert_eq!(highest_one_bit(0b1010), 0b1000);
/// ```
pub fn highest_one_bit(mut i: u64) -> u64 {
    i |= i >> 1;
    i |= i >> 2;
    i |= i >> 4;
    i |= i >> 8;
    i |= i >> 16;
    i |= i >> 32;
    i - (i >> 1)
}

/// Returns the value of the lowest one-bit in `i`.
///
/// # Examples
///
/// ```
/// use bc_rust::util::longs::lowest_one_bit;
/// assert_eq!(lowest_one_bit(0b1010), 0b0010);
/// ```
pub fn lowest_one_bit(i: u64) -> u64 {
    i & i.wrapping_neg()
}

/// Returns the number of leading zero bits in `i`.
///
/// Equivalent to Rust's built-in [`u64::leading_zeros`]. Prefer using it directly.
///
/// # Examples
///
/// ```
/// use bc_rust::util::longs::number_of_leading_zeros;
/// assert_eq!(number_of_leading_zeros(1), 63);
/// ```
pub fn number_of_leading_zeros(i: u64) -> u32 {
    i.leading_zeros()
}

/// Returns the number of trailing zero bits in `i`.
///
/// Equivalent to Rust's built-in [`u64::trailing_zeros`]. Prefer using it directly.
///
/// # Examples
///
/// ```
/// use bc_rust::util::longs::number_of_trailing_zeros;
/// assert_eq!(number_of_trailing_zeros(0b1000), 3);
/// ```
pub fn number_of_trailing_zeros(i: u64) -> u32 {
    i.trailing_zeros()
}

/// Returns the number of one-bits in `i` (population count).
///
/// Equivalent to Rust's built-in [`u64::count_ones`]. Prefer using it directly.
///
/// # Examples
///
/// ```
/// use bc_rust::util::longs::pop_count;
/// assert_eq!(pop_count(0b1010_1010), 4);
/// ```
pub fn pop_count(i: u64) -> u32 {
    i.count_ones()
}

/// Returns the value of `i` with its bits reversed.
///
/// Equivalent to Rust's built-in [`u64::reverse_bits`]. Prefer using it directly.
///
/// # Examples
///
/// ```
/// use bc_rust::util::longs::reverse;
/// assert_eq!(reverse(0x8000_0000_0000_0000), 0x0000_0000_0000_0001);
/// ```
pub fn reverse(i: u64) -> u64 {
    i.reverse_bits()
}

/// Returns the value of `i` with its bytes reversed.
///
/// Equivalent to Rust's built-in [`u64::swap_bytes`]. Prefer using it directly.
///
/// # Examples
///
/// ```
/// use bc_rust::util::longs::reverse_bytes;
/// assert_eq!(reverse_bytes(0x0102030405060708), 0x0807060504030201);
/// ```
pub fn reverse_bytes(i: u64) -> u64 {
    i.swap_bytes()
}

/// Returns the value of `i` rotated left by `distance` bits.
///
/// Equivalent to Rust's built-in [`u64::rotate_left`]. Prefer using it directly.
///
/// # Examples
///
/// ```
/// use bc_rust::util::longs::rotate_left;
/// assert_eq!(rotate_left(0x8000_0000_0000_0000, 1), 0x0000_0000_0000_0001);
/// ```
pub fn rotate_left(i: u64, distance: u32) -> u64 {
    i.rotate_left(distance)
}

/// Returns the value of `i` rotated right by `distance` bits.
///
/// Equivalent to Rust's built-in [`u64::rotate_right`]. Prefer using it directly.
///
/// # Examples
///
/// ```
/// use bc_rust::util::longs::rotate_right;
/// assert_eq!(rotate_right(0x0000_0000_0000_0001, 1), 0x8000_0000_0000_0000);
/// ```
pub fn rotate_right(i: u64, distance: u32) -> u64 {
    i.rotate_right(distance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highest_one_bit() {
        assert_eq!(highest_one_bit(0b0000), 0b0000);
        assert_eq!(highest_one_bit(0b0001), 0b0001);
        assert_eq!(highest_one_bit(0b1010), 0b1000);
        assert_eq!(highest_one_bit(0b1111), 0b1000);
    }

    #[test]
    fn test_lowest_one_bit() {
        assert_eq!(lowest_one_bit(0b0000), 0b0000);
        assert_eq!(lowest_one_bit(0b1000), 0b1000);
        assert_eq!(lowest_one_bit(0b1010), 0b0010);
        assert_eq!(lowest_one_bit(0b1111), 0b0001);
    }

    #[test]
    fn test_number_of_leading_zeros() {
        assert_eq!(number_of_leading_zeros(0), 64);
        assert_eq!(number_of_leading_zeros(1), 63);
        assert_eq!(number_of_leading_zeros(0x8000_0000_0000_0000), 0);
    }

    #[test]
    fn test_number_of_trailing_zeros() {
        assert_eq!(number_of_trailing_zeros(0), 64);
        assert_eq!(number_of_trailing_zeros(1), 0);
        assert_eq!(number_of_trailing_zeros(0b1000), 3);
    }

    #[test]
    fn test_pop_count() {
        assert_eq!(pop_count(0), 0);
        assert_eq!(pop_count(0b1010_1010), 4);
        assert_eq!(pop_count(u64::MAX), 64);
    }

    #[test]
    fn test_reverse() {
        assert_eq!(reverse(0x8000_0000_0000_0000), 0x0000_0000_0000_0001);
        assert_eq!(reverse(0x0000_0000_0000_0001), 0x8000_0000_0000_0000);
    }

    #[test]
    fn test_reverse_bytes() {
        assert_eq!(reverse_bytes(0x0102030405060708), 0x0807060504030201);
    }

    #[test]
    fn test_rotate_left() {
        assert_eq!(rotate_left(0b0001, 1), 0b0010);
        assert_eq!(rotate_left(0x8000_0000_0000_0000, 1), 0x0000_0000_0000_0001);
    }

    #[test]
    fn test_rotate_right() {
        assert_eq!(rotate_right(0b0010, 1), 0b0001);
        assert_eq!(
            rotate_right(0x0000_0000_0000_0001, 1),
            0x8000_0000_0000_0000
        );
    }
}
