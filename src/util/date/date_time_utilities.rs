use chrono::{self, TimeZone, Timelike};

use crate::{Error, ErrorKind, Result};

/// Return the number of milliseconds since the Unix epoch (1 Jan., 1970 UTC) for a given DateTime value.
/// # Arguments
/// * `date_time` - A DateTime value not before the epoch.
/// # Returns
/// Number of whole milliseconds after epoch.
/// # Errors
/// `date_time` value may not be before the epoch
/// # Remarks
/// The DateTime value will be converted to UTC (using `to_utc()` before conversion.
pub fn date_time_to_unix_ms<Tz: chrono::TimeZone>(date_time: &chrono::DateTime<Tz>) -> Result<i64> {
    let utc = date_time.to_utc();

    let result = utc.timestamp_millis();
    if result < 0 {
        return Err(Error::with_message(
            ErrorKind::InvalidInput,
            "date_time value may not be before the epoch".to_owned(),
        ));
    }
    Ok(result)
}

/// Create a `DateTime<Utc>` value from the number of milliseconds since the Unix epoch (1 Jan., 1970 UTC).
/// # Arguments
/// * `unix_ms` - Number of milliseconds since the epoch.
/// # Returns
///  `DateTime<Utc>` value
/// # Errors
/// `unix_ms` value may be out of range
pub fn unix_ms_to_date_time(unix_ms: i64) -> Result<chrono::DateTime<chrono::Utc>> {
    match chrono::Utc.timestamp_millis_opt(unix_ms) {
        chrono::MappedLocalTime::Single(result) => Ok(result),
        _ => Err(Error::with_message(
            ErrorKind::InvalidInput,
            "unix_ms out of range".to_owned(),
        )),
    }
}

/// Return the current number of milliseconds since the Unix epoch (1 Jan., 1970 UTC).
pub fn current_unix_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

const TICKS_PER_MILLI_SECOND: u32 = 10000;

/// Represents the number of ticks in 1 second.
const TICKS_PER_SENCOND: u32 = 10000000;

pub fn with_precision_centisecond<Tz: chrono::TimeZone>(
    date_time: &chrono::DateTime<Tz>,
) -> chrono::DateTime<Tz> {
    date_time
        .with_nanosecond(date_time.nanosecond() / (TICKS_PER_MILLI_SECOND * 10))
        .unwrap()
}

pub fn with_precision_decisecond<Tz: chrono::TimeZone>(
    date_time: &chrono::DateTime<Tz>,
) -> chrono::DateTime<Tz> {
    date_time
        .with_nanosecond(date_time.nanosecond() / (TICKS_PER_MILLI_SECOND * 100))
        .unwrap()
}

pub fn with_precision_millisecond<Tz: chrono::TimeZone>(
    date_time: &chrono::DateTime<Tz>,
) -> chrono::DateTime<Tz> {
    date_time
        .with_nanosecond(date_time.nanosecond() / TICKS_PER_MILLI_SECOND)
        .unwrap()
}

pub fn with_precision_second<Tz: chrono::TimeZone>(
    date_time: &chrono::DateTime<Tz>,
) -> chrono::DateTime<Tz> {
    date_time
        .with_nanosecond(date_time.nanosecond() / TICKS_PER_SENCOND)
        .unwrap()
}
