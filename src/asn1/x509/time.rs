use crate::asn1::{Asn1Convertible, Asn1GeneralizedTime, Asn1Object, Asn1UtcTime};
use crate::{Error, Result};
use anyhow::bail;
use chrono::{DateTime, Datelike, Local, TimeZone, Utc};
use std::fmt;
use std::fmt::Formatter;

/// Produce an object suitable for an Asn1OutputStream.
/// ```text
/// Time ::= CHOICE {
///     utcTime UTCTime,
///     generalTime GeneralizedTime 
/// }
/// ```
pub enum Time {
    GeneralizedTime(Asn1GeneralizedTime),
    UtcTime(Asn1UtcTime),
}

impl Time {
    pub fn with_date_time<Tz: TimeZone>(date_time: &DateTime<Tz>) -> Result<Time> {
        let utc = date_time.to_utc();
        if utc.year() < 1950 || utc.year() > 2049 {
            Ok(Time::GeneralizedTime(Asn1GeneralizedTime::with_date_time(&utc)))
        } else {
            Ok(Time::UtcTime(Asn1UtcTime::with_date_time(utc, 2049)?))
        }
    }
    pub fn with_asn1_generalized_time(generalized_time: Asn1GeneralizedTime) -> Self {
        Time::GeneralizedTime(generalized_time)
    }
    pub fn with_asn1_utc_time(utc_time: Asn1UtcTime) -> Result<Self> {
        utc_time.to_date_time(2049)?;
        Ok(Time::UtcTime(utc_time))
    }
    pub fn to_date_time(&self) -> Result<DateTime<Utc>> {
        match self {
            Time::UtcTime(ref utc_time) => utc_time.to_date_time(2049),
            Time::GeneralizedTime(ref generalized_time) => {
                Ok(generalized_time.to_date_time())
            }
            _ => bail!(Error::invalid_operation("invalid time object")),
        }
    }
}

impl fmt::Display for Time {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Time::UtcTime(ref utc_time) => {
                let date_time = utc_time
                    .to_date_time(2049)
                    .map_err(|_| fmt::Error::default())?;

                write!(f, "{}", date_time)
            }
            Time::GeneralizedTime(ref generalized_time) => {
                let date_time = generalized_time.to_date_time();
                write!(f, "{}", date_time)
            }
            _ => {
                write!(f, "Invalid time object")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_01() {}
}
