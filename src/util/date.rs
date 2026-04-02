//! Date and time utilities.
//!
//! Port of `DateTimeUtilities.cs` from bc-csharp.
//!
//! All timestamps are represented as `i64` milliseconds since the Unix epoch
//! (1 Jan 1970, 00:00:00 UTC). This is simpler than bc-csharp's `DateTime`/ticks
//! approach because Rust has no built-in calendar type.

use std::time::{Duration, SystemTime, UNIX_EPOCH};
use crate::error::{BcError, BcResult};

/// The minimum valid Unix timestamp in milliseconds (the Unix epoch itself).
///
/// Equivalent to bc-csharp's `DateTimeUtilities.MinUnixMs`.
pub const MIN_UNIX_MS: i64 = 0;

/// The maximum valid Unix timestamp in milliseconds.
///
/// Corresponds to 9999-12-31 23:59:59.999 UTC, matching bc-csharp's
/// `DateTimeUtilities.MaxUnixMs` which is derived from `DateTime.MaxValue`.
pub const MAX_UNIX_MS: i64 = 253_402_300_799_999;

/// Converts a [`SystemTime`] to milliseconds since the Unix epoch.
///
/// # Errors
///
/// Returns [`BcError::SystemTimeError`] if `time` is before the Unix epoch.
///
/// # Examples
///
/// ```
/// use std::time::{UNIX_EPOCH, Duration};
/// use bc_rust::util::date::to_unix_ms;
///
/// let t = UNIX_EPOCH + Duration::from_millis(1_000);
/// assert_eq!(to_unix_ms(t).unwrap(), 1_000);
/// ```
pub fn to_unix_ms(time: SystemTime) -> BcResult<i64> {
    let d = time.duration_since(UNIX_EPOCH)?;
    Ok(d.as_millis() as i64)
}

/// Creates a [`SystemTime`] from milliseconds since the Unix epoch.
///
/// # Errors
///
/// Returns [`BcError::InvalidArgument`] if `ms` is outside [`MIN_UNIX_MS`]..=[`MAX_UNIX_MS`].
///
/// # Examples
///
/// ```
/// use std::time::{UNIX_EPOCH, Duration};
/// use bc_rust::util::date::from_unix_ms;
///
/// let t = from_unix_ms(1_000).unwrap();
/// assert_eq!(t, UNIX_EPOCH + Duration::from_millis(1_000));
/// ```
pub fn from_unix_ms(ms: i64) -> BcResult<SystemTime> {
    if !(MIN_UNIX_MS..=MAX_UNIX_MS).contains(&ms) {
        return Err(BcError::InvalidArgument {
            param: Some("ms".to_string()),
            msg: format!("Unix millisecond value must be in {}..={}", MIN_UNIX_MS, MAX_UNIX_MS),
        });
    }
    Ok(UNIX_EPOCH + Duration::from_millis(ms as u64))
}

/// Returns the current number of milliseconds since the Unix epoch.
///
/// # Examples
///
/// ```
/// use bc_rust::util::date::current_unix_ms;
///
/// let ms = current_unix_ms();
/// assert!(ms > 0);
/// ```
pub fn current_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Truncates a Unix millisecond timestamp to centisecond precision (10 ms).
///
/// # Examples
///
/// ```
/// use bc_rust::util::date::with_precision_centisecond;
/// assert_eq!(with_precision_centisecond(1_234), 1_230);
/// ```
pub fn with_precision_centisecond(ms: i64) -> i64 {
    ms / 10 * 10
}

/// Truncates a Unix millisecond timestamp to decisecond precision (100 ms).
///
/// # Examples
///
/// ```
/// use bc_rust::util::date::with_precision_decisecond;
/// assert_eq!(with_precision_decisecond(1_234), 1_200);
/// ```
pub fn with_precision_decisecond(ms: i64) -> i64 {
    ms / 100 * 100
}

/// Truncates a Unix millisecond timestamp to millisecond precision.
///
/// This is a no-op since the input is already in milliseconds.
/// Included for API symmetry with the other `with_precision_*` functions.
///
/// # Examples
///
/// ```
/// use bc_rust::util::date::with_precision_millisecond;
/// assert_eq!(with_precision_millisecond(1_234), 1_234);
/// ```
pub fn with_precision_millisecond(ms: i64) -> i64 {
    ms
}

/// Truncates a Unix millisecond timestamp to second precision (1000 ms).
///
/// # Examples
///
/// ```
/// use bc_rust::util::date::with_precision_second;
/// assert_eq!(with_precision_second(1_234), 1_000);
/// ```
pub fn with_precision_second(ms: i64) -> i64 {
    ms / 1_000 * 1_000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_unix_ms() {
        let t = UNIX_EPOCH + Duration::from_millis(1_000);
        assert_eq!(to_unix_ms(t).unwrap(), 1_000);
    }

    #[test]
    fn test_to_unix_ms_epoch() {
        assert_eq!(to_unix_ms(UNIX_EPOCH).unwrap(), 0);
    }

    #[test]
    fn test_to_unix_ms_before_epoch() {
        let t = UNIX_EPOCH - Duration::from_millis(1);
        assert!(to_unix_ms(t).is_err());
    }

    #[test]
    fn test_from_unix_ms() {
        let t = from_unix_ms(1_000).unwrap();
        assert_eq!(t, UNIX_EPOCH + Duration::from_millis(1_000));
    }

    #[test]
    fn test_from_unix_ms_zero() {
        assert_eq!(from_unix_ms(0).unwrap(), UNIX_EPOCH);
    }

    #[test]
    fn test_from_unix_ms_negative() {
        assert!(from_unix_ms(-1).is_err());
    }

    #[test]
    fn test_from_unix_ms_max() {
        assert!(from_unix_ms(MAX_UNIX_MS).is_ok());
        assert!(from_unix_ms(MAX_UNIX_MS + 1).is_err());
    }

    #[test]
    fn test_current_unix_ms() {
        let ms = current_unix_ms();
        assert!(ms > 0);
    }

    #[test]
    fn test_with_precision_second() {
        assert_eq!(with_precision_second(1_234), 1_000);
        assert_eq!(with_precision_second(1_000), 1_000);
        assert_eq!(with_precision_second(999), 0);
    }

    #[test]
    fn test_with_precision_decisecond() {
        assert_eq!(with_precision_decisecond(1_234), 1_200);
        assert_eq!(with_precision_decisecond(1_100), 1_100);
    }

    #[test]
    fn test_with_precision_centisecond() {
        assert_eq!(with_precision_centisecond(1_234), 1_230);
        assert_eq!(with_precision_centisecond(1_230), 1_230);
    }

    #[test]
    fn test_with_precision_millisecond() {
        assert_eq!(with_precision_millisecond(1_234), 1_234);
    }
}
