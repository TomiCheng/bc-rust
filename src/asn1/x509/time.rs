use crate::Result;
use crate::asn1::{Asn1GeneralizedTime, Asn1Object, Asn1UtcTime};
use chrono::{DateTime, Datelike, TimeZone, Utc};
use std::fmt;
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
    pub fn from_asn1_generalized_time(generalized_time: Asn1GeneralizedTime) -> Self {
        Time::GeneralizedTime(generalized_time)
    }
    pub fn from_asn1_utc_time(utc_time: Asn1UtcTime) -> Result<Self> {
        utc_time.to_date_time(2049)?;
        Ok(Time::UtcTime(utc_time))
    }
    /// Return our time as DateTime.
    pub fn to_date_time(&self) -> Result<DateTime<Utc>> {
        match self {
            Time::UtcTime(utc_time) => utc_time.to_date_time(2049),
            Time::GeneralizedTime(generalized_time) => Ok(generalized_time.to_date_time()),
        }
    }
}
impl From<Time> for Asn1Object {
    fn from(value: Time) -> Self {
        match value {
            Time::UtcTime(utc_time) => Asn1Object::UtcTime(utc_time),
            Time::GeneralizedTime(generalized_time) => Asn1Object::GeneralizedTime(generalized_time),
        }
    }
}
impl TryFrom<Asn1Object> for Time {
    type Error = crate::BcError;

    fn try_from(value: Asn1Object) -> Result<Self> {
        match value {
            Asn1Object::UtcTime(utc_time) => Ok(Time::from_asn1_utc_time(utc_time)?),
            Asn1Object::GeneralizedTime(generalized_time) => Ok(Time::from_asn1_generalized_time(generalized_time)),
            _ => Err(crate::BcError::with_invalid_format("expected Asn1Object to be a time")),
        }
    }
}
impl fmt::Display for Time {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Time::UtcTime(utc_time) => {
                let date_time = utc_time.to_date_time(2049).map_err(|_| fmt::Error::default())?;

                write!(f, "{}", date_time)
            }
            Time::GeneralizedTime(generalized_time) => {
                let date_time = generalized_time.to_date_time();
                write!(f, "{}", date_time)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_01() {}
}
