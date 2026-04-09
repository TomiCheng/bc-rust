//! Array utility functions.
//!
//! Port of `crypto/src/util/Arrays.cs` from bc-csharp.
//!
//! Rust's standard library already covers many array operations (equality via `==`,
//! cloning via `.to_vec()`, searching via `.contains()`, etc.). This module provides
//! only the functions that have no direct standard-library equivalent, plus
//! security-sensitive operations that require special handling.
//!
//! # Implemented Functions
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`are_all_zeroes`] | Check whether every byte in a slice is zero |
//! | [`fixed_time_equals`] | Constant-time byte-slice comparison (timing-attack safe) |
//! | [`copy_of`] | Copy a slice, truncating or zero-padding to a new length |
//! | [`copy_of_range`] | Copy a sub-range, zero-padding if the range exceeds the slice |
//! | [`concatenate`] | Concatenate two byte slices into a new `Vec<u8>` |
//! | [`concatenate_all`] | Concatenate any number of byte slices into a new `Vec<u8>` |
//! | [`zero_memory`] | Securely zero a mutable byte slice (optimizer-resistant) |
//! | [`segments_overlap`] | Determine whether two index ranges overlap |
//!
//! # Not Implemented — Rust Standard Library Equivalents
//!
//! The following bc-csharp functions are covered by Rust's standard library
//! and do not require a separate implementation:
//!
//! | bc-csharp | Rust equivalent |
//! |-----------|-----------------|
//! | `AreEqual(T[], T[])` | `a == b` (`PartialEq`) |
//! | `AreEqual(T[], from, to, T[], from, to)` | `a[from..to] == b[from..to]` |
//! | `Clone(data)` | `data.to_vec()` |
//! | `Clone(data, existing)` | `existing.copy_from_slice(data)` |
//! | `Contains(a, n)` | `a.contains(&n)` |
//! | `Fill(buf, value)` | `buf.fill(value)` |
//! | `Fill(buf, from, to, value)` | `buf[from..to].fill(value)` |
//! | `Clear(buf)` | `buf.fill(0)` |
//! | `Append(a, b)` | `vec.push(b)` |
//! | `Prepend(a, b)` | `vec.insert(0, b)` |
//! | `Reverse(a)` | `a.iter().rev().cloned().collect()` |
//! | `ReverseInPlace(a)` | `a.reverse()` |
//! | `IsNullOrEmpty(a)` | `a.is_empty()` |
//! | `GetHashCode(data)` | `#[derive(Hash)]` / `std::hash::Hash` |
//! | `ValidateBuffer` / `ValidateSegment` / `ValidateRange` | Rust slice indexing panics on out-of-bounds |
//! | `CopyBuffer` / `CopySegment` | `&data[off..off+len].to_vec()` |

/// Returns `true` if every byte in `buf` is zero.
///
/// # Examples
///
/// ```
/// use bc_rust::util::arrays::are_all_zeroes;
///
/// assert!(are_all_zeroes(&[0, 0, 0]));
/// assert!(!are_all_zeroes(&[0, 1, 0]));
/// ```
pub fn are_all_zeroes(buf: &[u8]) -> bool {
    let mut bits: u8 = 0;
    for &b in buf {
        bits |= b;
    }
    bits == 0
}

/// Compares two byte slices in **constant time**.
///
/// Unlike `a == b`, this function always iterates over all bytes regardless of
/// where a difference occurs, preventing timing-based side-channel attacks.
///
/// Returns `false` immediately (without scanning) only when the slice lengths differ,
/// because the length itself is not considered secret in typical protocols.
///
/// # Examples
///
/// ```
/// use bc_rust::util::arrays::fixed_time_equals;
///
/// assert!(fixed_time_equals(b"hello", b"hello"));
/// assert!(!fixed_time_equals(b"hello", b"world"));
/// assert!(!fixed_time_equals(b"hi", b"hello"));
/// ```
#[inline(never)]
pub fn fixed_time_equals(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Returns a new `Vec<u8>` of exactly `new_length` bytes.
///
/// - If `new_length` is shorter than `data`, the result is truncated.
/// - If `new_length` is longer than `data`, the extra bytes are zero-padded.
///
/// # Examples
///
/// ```
/// use bc_rust::util::arrays::copy_of;
///
/// assert_eq!(copy_of(&[1, 2, 3, 4], 2), vec![1, 2]);
/// assert_eq!(copy_of(&[1, 2], 4), vec![1, 2, 0, 0]);
/// ```
pub fn copy_of(data: &[u8], new_length: usize) -> Vec<u8> {
    let mut result = vec![0u8; new_length];
    let copy_len = new_length.min(data.len());
    result[..copy_len].copy_from_slice(&data[..copy_len]);
    result
}

/// Copies bytes from `data[from..to]` into a new `Vec<u8>`.
///
/// The range `[from, to)` may extend beyond the end of `data`; any bytes
/// past the end of `data` are filled with zeros.
///
/// # Panics
///
/// Panics if `from > to`.
///
/// # Examples
///
/// ```
/// use bc_rust::util::arrays::copy_of_range;
///
/// assert_eq!(copy_of_range(&[1, 2, 3, 4], 1, 3), vec![2, 3]);
/// // Range extends beyond end — extra bytes are zero-padded
/// assert_eq!(copy_of_range(&[1, 2, 3], 1, 5), vec![2, 3, 0, 0]);
/// ```
pub fn copy_of_range(data: &[u8], from: usize, to: usize) -> Vec<u8> {
    assert!(from <= to, "from ({from}) > to ({to})");
    let new_length = to - from;
    let mut result = vec![0u8; new_length];
    let available = data.len().saturating_sub(from);
    let copy_len = new_length.min(available);
    result[..copy_len].copy_from_slice(&data[from..from + copy_len]);
    result
}

/// Concatenates two byte slices into a new `Vec<u8>`.
///
/// # Examples
///
/// ```
/// use bc_rust::util::arrays::concatenate;
///
/// assert_eq!(concatenate(b"hello", b" world"), b"hello world");
/// assert_eq!(concatenate(b"", b"abc"), b"abc");
/// ```
pub fn concatenate(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    result.extend_from_slice(a);
    result.extend_from_slice(b);
    result
}

/// Concatenates any number of byte slices into a new `Vec<u8>`.
///
/// # Examples
///
/// ```
/// use bc_rust::util::arrays::concatenate_all;
///
/// assert_eq!(concatenate_all(&[b"foo", b"bar", b"baz"]), b"foobarbaz");
/// assert_eq!(concatenate_all(&[]), b"");
/// ```
pub fn concatenate_all(slices: &[&[u8]]) -> Vec<u8> {
    let total = slices.iter().map(|s| s.len()).sum();
    let mut result = Vec::with_capacity(total);
    for s in slices {
        result.extend_from_slice(s);
    }
    result
}

/// Securely zeros every byte in `buf`, resisting compiler optimizations that
/// might otherwise eliminate the writes as "dead stores".
///
/// # Examples
///
/// ```
/// use bc_rust::util::arrays::zero_memory;
///
/// let mut secret = vec![1u8, 2, 3, 4];
/// zero_memory(&mut secret);
/// assert_eq!(secret, vec![0, 0, 0, 0]);
/// ```
pub fn zero_memory(buf: &mut [u8]) {
    for b in buf.iter_mut() {
        // SAFETY: writing through a valid mutable reference.
        unsafe { std::ptr::write_volatile(b, 0) };
    }
}

/// Returns `true` if the two index ranges `[a_off, a_off + a_len)` and
/// `[b_off, b_off + b_len)` overlap.
///
/// Empty ranges (length 0) never overlap.
///
/// # Examples
///
/// ```
/// use bc_rust::util::arrays::segments_overlap;
///
/// assert!(segments_overlap(0, 4, 2, 4));  // [0,4) and [2,6) overlap
/// assert!(!segments_overlap(0, 2, 2, 2)); // [0,2) and [2,4) are adjacent, not overlapping
/// assert!(!segments_overlap(0, 0, 0, 4)); // empty range never overlaps
/// ```
pub fn segments_overlap(a_off: usize, a_len: usize, b_off: usize, b_len: usize) -> bool {
    if a_len == 0 || b_len == 0 {
        return false;
    }
    a_off < b_off + b_len && b_off < a_off + a_len
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- are_all_zeroes ---

    #[test]
    fn test_are_all_zeroes_true() {
        assert!(are_all_zeroes(&[0, 0, 0]));
    }

    #[test]
    fn test_are_all_zeroes_false() {
        assert!(!are_all_zeroes(&[0, 0, 1]));
    }

    #[test]
    fn test_are_all_zeroes_empty() {
        assert!(are_all_zeroes(&[]));
    }

    // --- fixed_time_equals ---

    #[test]
    fn test_fixed_time_equals_equal() {
        assert!(fixed_time_equals(b"hello", b"hello"));
    }

    #[test]
    fn test_fixed_time_equals_different_content() {
        assert!(!fixed_time_equals(b"hello", b"world"));
    }

    #[test]
    fn test_fixed_time_equals_different_length() {
        assert!(!fixed_time_equals(b"hi", b"hello"));
    }

    #[test]
    fn test_fixed_time_equals_empty() {
        assert!(fixed_time_equals(b"", b""));
    }

    // --- copy_of ---

    #[test]
    fn test_copy_of_truncate() {
        assert_eq!(copy_of(&[1, 2, 3, 4], 2), vec![1, 2]);
    }

    #[test]
    fn test_copy_of_pad() {
        assert_eq!(copy_of(&[1, 2], 4), vec![1, 2, 0, 0]);
    }

    #[test]
    fn test_copy_of_same_length() {
        assert_eq!(copy_of(&[1, 2, 3], 3), vec![1, 2, 3]);
    }

    #[test]
    fn test_copy_of_zero_length() {
        assert_eq!(copy_of(&[1, 2, 3], 0), vec![]);
    }

    // --- copy_of_range ---

    #[test]
    fn test_copy_of_range_basic() {
        assert_eq!(copy_of_range(&[1, 2, 3, 4], 1, 3), vec![2, 3]);
    }

    #[test]
    fn test_copy_of_range_past_end() {
        assert_eq!(copy_of_range(&[1, 2, 3], 1, 5), vec![2, 3, 0, 0]);
    }

    #[test]
    fn test_copy_of_range_full() {
        assert_eq!(copy_of_range(&[1, 2, 3], 0, 3), vec![1, 2, 3]);
    }

    #[test]
    #[should_panic]
    fn test_copy_of_range_invalid() {
        copy_of_range(&[1, 2, 3], 3, 1);
    }

    // --- concatenate ---

    #[test]
    fn test_concatenate_basic() {
        assert_eq!(concatenate(b"hello", b" world"), b"hello world");
    }

    #[test]
    fn test_concatenate_empty_left() {
        assert_eq!(concatenate(b"", b"abc"), b"abc");
    }

    #[test]
    fn test_concatenate_empty_right() {
        assert_eq!(concatenate(b"abc", b""), b"abc");
    }

    #[test]
    fn test_concatenate_both_empty() {
        assert_eq!(concatenate(b"", b""), b"");
    }

    // --- concatenate_all ---

    #[test]
    fn test_concatenate_all_basic() {
        assert_eq!(concatenate_all(&[b"foo", b"bar", b"baz"]), b"foobarbaz");
    }

    #[test]
    fn test_concatenate_all_empty_slices() {
        assert_eq!(concatenate_all(&[]), b"");
    }

    #[test]
    fn test_concatenate_all_single() {
        assert_eq!(concatenate_all(&[b"abc"]), b"abc");
    }

    // --- zero_memory ---

    #[test]
    fn test_zero_memory() {
        let mut buf = vec![1u8, 2, 3, 4];
        zero_memory(&mut buf);
        assert_eq!(buf, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_zero_memory_empty() {
        let mut buf: Vec<u8> = vec![];
        zero_memory(&mut buf);
        assert!(buf.is_empty());
    }

    // --- segments_overlap ---

    #[test]
    fn test_segments_overlap_true() {
        assert!(segments_overlap(0, 4, 2, 4)); // [0,4) and [2,6)
    }

    #[test]
    fn test_segments_overlap_adjacent() {
        assert!(!segments_overlap(0, 2, 2, 2)); // [0,2) and [2,4) adjacent
    }

    #[test]
    fn test_segments_overlap_no_overlap() {
        assert!(!segments_overlap(0, 2, 5, 3)); // [0,2) and [5,8)
    }

    #[test]
    fn test_segments_overlap_empty_a() {
        assert!(!segments_overlap(0, 0, 0, 4));
    }

    #[test]
    fn test_segments_overlap_empty_b() {
        assert!(!segments_overlap(0, 4, 0, 0));
    }

    #[test]
    fn test_segments_overlap_contained() {
        assert!(segments_overlap(0, 10, 2, 3)); // [0,10) contains [2,5)
    }
}
