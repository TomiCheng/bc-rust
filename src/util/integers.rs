//! Utility functions for 32-bit integer bit manipulation.

pub const NUM_BITS: u32 = 32;
pub const NUM_BYTES: u32 = 4;

/// Returns the value of the highest one-bit in `i`.
///
/// # Examples
///
/// ```
/// use bc_rust::util::integers::highest_one_bit;
/// assert_eq!(highest_one_bit(0b1010), 0b1000);
/// ```
pub fn highest_one_bit(mut i: u32) -> u32 {
    i |= i >> 1;
    i |= i >> 2;
    i |= i >> 4;
    i |= i >> 8;
    i |= i >> 16;
    i - (i >> 1)
}

/// Returns the value of the lowest one-bit in `i`.
///
/// # Examples
///
/// ```
/// use bc_rust::util::integers::lowest_one_bit;
/// assert_eq!(lowest_one_bit(0b1010), 0b0010);
/// ```
pub fn lowest_one_bit(i: u32) -> u32 {
    i & i.wrapping_neg()
}

/// Returns the number of leading zero bits in `i`.
///
/// Equivalent to Rust's built-in [`u32::leading_zeros`]. Prefer using it directly.
///
/// # Examples
///
/// ```
/// use bc_rust::util::integers::number_of_leading_zeros;
/// assert_eq!(number_of_leading_zeros(1), 31);
/// ```
pub fn number_of_leading_zeros(i: u32) -> u32 {
    i.leading_zeros()
}

/// Returns the number of trailing zero bits in `i`.
///
/// Equivalent to Rust's built-in [`u32::trailing_zeros`]. Prefer using it directly.
///
/// # Examples
///
/// ```
/// use bc_rust::util::integers::number_of_trailing_zeros;
/// assert_eq!(number_of_trailing_zeros(0b1000), 3);
/// ```
pub fn number_of_trailing_zeros(i: u32) -> u32 {
    i.trailing_zeros()
}

/// Returns the number of one-bits in `i` (population count).
///
/// Equivalent to Rust's built-in [`u32::count_ones`]. Prefer using it directly.
///
/// # Examples
///
/// ```
/// use bc_rust::util::integers::pop_count;
/// assert_eq!(pop_count(0b1010_1010), 4);
/// ```
pub fn pop_count(i: u32) -> u32 {
    i.count_ones()
}

/// Returns the value of `i` with its bits reversed.
///
/// Equivalent to Rust's built-in [`u32::reverse_bits`]. Prefer using it directly.
///
/// # Examples
///
/// ```
/// use bc_rust::util::integers::reverse;
/// assert_eq!(reverse(0x8000_0000), 0x0000_0001);
/// ```
pub fn reverse(i: u32) -> u32 {
    i.reverse_bits()
}

/// Returns the value of `i` with its bytes reversed.
///
/// Equivalent to Rust's built-in [`u32::swap_bytes`]. Prefer using it directly.
///
/// # Examples
///
/// ```
/// use bc_rust::util::integers::reverse_bytes;
/// assert_eq!(reverse_bytes(0x12345678), 0x78563412);
/// ```
pub fn reverse_bytes(i: u32) -> u32 {
    i.swap_bytes()
}

/// Returns the value of `i` rotated left by `distance` bits.
///
/// Equivalent to Rust's built-in [`u32::rotate_left`]. Prefer using it directly.
///
/// # Examples
///
/// ```
/// use bc_rust::util::integers::rotate_left;
/// assert_eq!(rotate_left(0x8000_0000, 1), 0x0000_0001);
/// ```
pub fn rotate_left(i: u32, distance: u32) -> u32 {
    i.rotate_left(distance)
}

/// Returns the value of `i` rotated right by `distance` bits.
///
/// Equivalent to Rust's built-in [`u32::rotate_right`]. Prefer using it directly.
///
/// # Examples
///
/// ```
/// use bc_rust::util::integers::rotate_right;
/// assert_eq!(rotate_right(0x0000_0001, 1), 0x8000_0000);
/// ```
pub fn rotate_right(i: u32, distance: u32) -> u32 {
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
        assert_eq!(number_of_leading_zeros(0), 32);
        assert_eq!(number_of_leading_zeros(1), 31);
        assert_eq!(number_of_leading_zeros(0x8000_0000), 0);
    }

    #[test]
    fn test_number_of_trailing_zeros() {
        assert_eq!(number_of_trailing_zeros(0), 32);
        assert_eq!(number_of_trailing_zeros(1), 0);
        assert_eq!(number_of_trailing_zeros(0b1000), 3);
    }

    #[test]
    fn test_pop_count() {
        assert_eq!(pop_count(0), 0);
        assert_eq!(pop_count(0b1010_1010), 4);
        assert_eq!(pop_count(u32::MAX), 32);
    }

    #[test]
    fn test_reverse() {
        assert_eq!(reverse(0b1000_0000_0000_0000_0000_0000_0000_0000), 0b0000_0001);
        assert_eq!(reverse(0b0000_0001), 0b1000_0000_0000_0000_0000_0000_0000_0000);
    }

    #[test]
    fn test_reverse_bytes() {
        assert_eq!(reverse_bytes(0x12345678), 0x78563412);
    }

    #[test]
    fn test_rotate_left() {
        assert_eq!(rotate_left(0b0001, 1), 0b0010);
        assert_eq!(rotate_left(0x8000_0000, 1), 0x0000_0001);
    }

    #[test]
    fn test_rotate_right() {
        assert_eq!(rotate_right(0b0010, 1), 0b0001);
        assert_eq!(rotate_right(0x0000_0001, 1), 0x8000_0000);
    }
}
