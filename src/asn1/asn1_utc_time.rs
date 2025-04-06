use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_tags::{UNIVERSAL, UTC_TIME};
use crate::asn1::asn1_write::{get_encoding_type, EncodingType};
use crate::asn1::primitive_encoding::PrimitiveEncoding;
use crate::asn1::{Asn1Encodable, Asn1Write};
use crate::{Error, Result};
use anyhow::{bail, ensure};
use chrono::{DateTime, Datelike, NaiveDateTime, SubsecRound, TimeZone, Utc};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::io::Write;

/// Represents an ASN.1 UTC Time value.
#[derive(Debug, Clone)]
pub struct Asn1UtcTime {
    date_time: DateTime<Utc>,
    date_time_locked: bool,
    date_time_string: String,
}

impl Asn1UtcTime {
    pub fn with_str(v: &str) -> Result<Asn1UtcTime> {
        let date_time = from_str(v)?;
        Ok(Asn1UtcTime {
            date_time,
            date_time_locked: false,
            date_time_string: to_string_canonical(&date_time),
        })
    }
    pub(crate) fn with_vec(v: Vec<u8>) -> Result<Self> {
        let ss = String::from_utf8(v)?;
        Self::with_str(&ss)
    }
    /// Creates a new `Asn1UtcTime` instance with the given date and time.
    /// # Arguments
    /// - `date_time`: A `DateTime<Tz: TimeZone>` representing the date and time.
    /// # Returns
    /// - A new `Asn1UtcTime` instance.
    /// # Errors
    /// - Returns an error if the year is out of `two_digit_year_max`.
    /// # Example
    /// ```rust
    /// use chrono::{DateTime, Utc};
    /// use bc_rust::asn1::Asn1UtcTime;
    /// let date_time = Utc::now();
    /// let asn1_utc_time = Asn1UtcTime::with_date_time(date_time, 2049).unwrap();
    /// let date_time1 = asn1_utc_time.to_date_time(2049).unwrap();
    /// assert_eq!(date_time, date_time1);
    /// ```
    ///
    pub fn with_date_time<Tz: TimeZone>(
        date_time: DateTime<Tz>,
        two_digit_year_max: i32,
    ) -> Result<Self> {
        let utc_date_time = date_time.to_utc();
        let utc_date_time = utc_date_time.trunc_subsecs(0);

        validate(&utc_date_time, two_digit_year_max)?;

        let date_time_string = to_string_canonical(&utc_date_time);
        Ok(Asn1UtcTime {
            date_time: utc_date_time,
            date_time_locked: true,
            date_time_string,
        })
    }

    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        Self::with_vec(contents)
    }
    pub fn to_date_time(&self, max_two_digit_year: i32) -> Result<DateTime<Utc>> {
        let date_time = self.date_time;
        if in_range(&date_time, max_two_digit_year) {
            return Ok(date_time);
        }
        ensure!(self.date_time_locked, Error::invalid_operation(""));
        let two_digit_year = date_time.year() % 100;
        let two_digit_year_cutoff = max_two_digit_year % 100;
        let diff = two_digit_year - two_digit_year_cutoff;
        let mut new_year = max_two_digit_year + diff;
        if diff > 100 {
            new_year -= 100;
        }
        let result = date_time.with_year(new_year);
        if let Some(result) = result {
            Ok(result)
        } else {
            bail!(Error::invalid_operation("invalid date"))
        }
    }
    fn get_encoding_with_type(&self, encode_type: EncodingType) -> impl Asn1Encoding {
        let mut contents = self.date_time_string.clone().into_bytes();
        if encode_type == EncodingType::Der {
            let canonical = to_string_canonical(&self.date_time);
            contents = canonical.into_bytes();
        }
        PrimitiveEncoding::new(UNIVERSAL, UTC_TIME, contents)
    }
}

impl PartialEq for Asn1UtcTime {
    fn eq(&self, other: &Self) -> bool {
        self.date_time == other.date_time
    }
}

impl Hash for Asn1UtcTime {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.date_time.hash(state);
    }
}

impl fmt::Display for Asn1UtcTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.date_time_string)
    }
}

impl Asn1Encodable for Asn1UtcTime {
    fn encode_to_with_encoding(&self, writer: &mut dyn Write, encoding_str: &str) -> Result<usize> {
        let encode_type = get_encoding_type(encoding_str);
        let encoding = self.get_encoding_with_type(encode_type);
        let mut asn1_writer = Asn1Write::create_with_encoding(writer, encoding_str);
        encoding.encode(&mut asn1_writer)
    }
}

fn from_str(s: &str) -> Result<DateTime<Utc>> {
    match s.len() {
        11 => parse_utc(s, "%Y%m%d%H%M%Z"),
        13 => parse_utc(s, "%Y%m%d%H%M%S%Z"),
        15 => parse_time_zone(s, "%Y%m%d%H%M%S%#z"),
        17 => parse_time_zone(s, "%Y%m%d%H%M%S%z"),
        _ => bail!(Error::invalid_format("invalid length")),
    }
}

fn parse_utc(s: &str, fmt: &str) -> Result<DateTime<Utc>> {
    let mut year = i32::from_str_radix(&s[0..2], 10)?;
    if year < 50 {
        year += 2000;
    } else {
        year += 1900;
    }
    let ss = format!("{}{}", year, &s[2..]);
    let local_date_time = NaiveDateTime::parse_from_str(&ss, fmt)?;
    let date_time = local_date_time.and_local_timezone(Utc).unwrap();
    Ok(date_time)
}
fn parse_time_zone(s: &str, fmt: &str) -> Result<DateTime<Utc>> {
    let mut year = i32::from_str_radix(&s[0..2], 10)?;
    if year < 50 {
        year += 2000;
    } else {
        year += 1900;
    }
    let ss = format!("{}{}", year, &s[2..]);
    let local_date_time = DateTime::parse_from_str(&ss, fmt)?;
    let date_time = local_date_time.with_timezone(&Utc);
    Ok(date_time)
}
fn in_range(date_time: &DateTime<Utc>, max_two_digit_year: i32) -> bool {
    (max_two_digit_year - date_time.year()) < 100
}

fn to_string_canonical(date_time: &DateTime<Utc>) -> String {
    date_time.format("%y%m%d%H%M%SZ").to_string()
}

fn validate(date_time: &DateTime<Utc>, two_digit_year_max: i32) -> Result<()> {
    if in_range(date_time, two_digit_year_max) {
        return Ok(());
    }
    bail!(Error::invalid_argument("out of range", "date_time"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Timelike;
    #[test]
    fn test_with_date_time() {
        let date_time = Utc::now();
        let asn1_utc_time = Asn1UtcTime::with_date_time(date_time, 2049).unwrap();
        let date_time1 = asn1_utc_time.to_date_time(2049).unwrap();
        assert_eq!(date_time.trunc_subsecs(0), date_time1);
    }
    #[test]
    fn test_with_str_01() {
        let str = "2301011234Z";
        let asn1_utc_time = Asn1UtcTime::with_str(str).unwrap();
        let date_time = asn1_utc_time.to_date_time(2049).unwrap();
        assert_eq!(date_time.year(), 2023);
        assert_eq!(date_time.month(), 1);
        assert_eq!(date_time.day(), 1);
        assert_eq!(date_time.hour(), 12);
        assert_eq!(date_time.minute(), 34);
    }
    #[test]
    fn test_with_str_02() {
        let str = "230101123456Z";
        let asn1_utc_time = Asn1UtcTime::with_str(str).unwrap();
        let date_time = asn1_utc_time.to_date_time(2049).unwrap();
        assert_eq!(date_time.year(), 2023);
        assert_eq!(date_time.month(), 1);
        assert_eq!(date_time.day(), 1);
        assert_eq!(date_time.hour(), 12);
        assert_eq!(date_time.minute(), 34);
        assert_eq!(date_time.second(), 56);
    }
    #[test]
    fn test_with_str_03() {
        let str = "230101123456+08";
        let asn1_utc_time = Asn1UtcTime::with_str(str).unwrap();
        let date_time = asn1_utc_time.to_date_time(2049).unwrap();
        assert_eq!(date_time.year(), 2023);
        assert_eq!(date_time.month(), 1);
        assert_eq!(date_time.day(), 1);
        assert_eq!(date_time.hour(), 12 - 8);
        assert_eq!(date_time.minute(), 34);
        assert_eq!(date_time.second(), 56);
    }
    #[test]
    fn test_with_str_04() {
        let str = "230101123456+0800";
        let asn1_utc_time = Asn1UtcTime::with_str(str).unwrap();
        let date_time = asn1_utc_time.to_date_time(2049).unwrap();
        assert_eq!(date_time.year(), 2023);
        assert_eq!(date_time.month(), 1);
        assert_eq!(date_time.day(), 1);
        assert_eq!(date_time.hour(), 12 - 8);
        assert_eq!(date_time.minute(), 34);
        assert_eq!(date_time.second(), 56);
    }

    #[test]
    fn test_to_string_canonical() {
        let str = "230101123456Z";
        let asn1_utc_time = Asn1UtcTime::with_str(str).unwrap();
        let date_time = asn1_utc_time.to_date_time(2049).unwrap();
        let canonical_str = to_string_canonical(&date_time);
        assert_eq!(canonical_str, "230101123456Z");
    }
}
