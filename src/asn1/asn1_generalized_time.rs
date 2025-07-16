use crate::{BcError, Result};
use chrono::{DateTime, Local, NaiveDateTime, TimeZone, Utc};
use std::fmt;
use std::hash::Hash;
use crate::asn1::asn1_encodable::Asn1EncodingInternal;

/// GeneralizedTime ASN.1 type
#[derive(Clone, Debug, Eq)]
pub struct Asn1GeneralizedTime {
    date_time: DateTime<Utc>,
    time_string_canonical: bool,
    time_string: String,
}
impl Asn1GeneralizedTime {
    /// Create a new instance of Asn1GeneralizedTime from a DateTime<Tz>
    /// # Arguments
    /// * `date_time` - A DateTime<Tz> representing the time
    /// # Returns
    /// A new instance of Asn1GeneralizedTime
    /// # Example
    /// ```
    /// use chrono::prelude::*;
    /// use bc_rust::asn1;
    /// let time = Utc.with_ymd_and_hms(2021, 1, 1, 0, 0, 0).unwrap();
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_date_time(&time).date_time());
    /// ```
    pub fn with_date_time<Tz: TimeZone>(date_time: &DateTime<Tz>) -> Self {
        Asn1GeneralizedTime {
            date_time: date_time.with_timezone(&Utc),
            time_string_canonical: true,
            time_string: to_string_canonical(&date_time.with_timezone(&Utc)),
        }
    }
    /// Create a new instance of Asn1GeneralizedTime from a [`str`]
    /// # Arguments
    /// * `time_string` - A string representing the time
    /// # Returns
    /// A new instance of Asn1GeneralizedTime
    /// # Errors
    /// Returns an error if the string is not a valid time string
    /// # Example
    /// ```
    /// use chrono::prelude::*;
    /// use bc_rust::asn1;
    /// // UTC
    /// let time = Utc.with_ymd_and_hms(2021, 1, 1, 0, 0, 0).unwrap();
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("2021010100Z").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("202101010000Z").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000Z").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.0Z").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.00Z").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.000Z").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.0000Z").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.00000Z").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.000000Z").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.0000000Z").unwrap().date_time());
    /// // Local
    /// let time = Local.with_ymd_and_hms(2021, 1, 1, 0, 0, 0).unwrap().to_utc();
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("2021010100").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("202101010000").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.0").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.00").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.000").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.0000").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.00000").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.000000").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.0000000").unwrap().date_time());
    /// // Time zone
    /// let time = Utc.with_ymd_and_hms(2020, 12, 31, 23, 0, 0).unwrap();
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("2021010100+0100").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("202101010000+0100").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000+0100").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.0+0100").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.00+0100").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.000+0100").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.0000+0100").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.00000+0100").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.000000+0100").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.0000000+0100").unwrap().date_time());
    ///
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("2021010100+01").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("202101010000+01").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000+01").unwrap().date_time());
    /// assert_eq!(&time, asn1::Asn1GeneralizedTime::with_str("20210101000000.0+01").unwrap().date_time());
    /// ```
    pub fn with_str(time_string: &str) -> Result<Self> {
        let date_time = from_str(time_string)?;
        Ok(Asn1GeneralizedTime {
            date_time,
            time_string_canonical: false,
            time_string: time_string.to_string(),
        })
    }

    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        let time_string = String::from_utf8(contents)?;
        let date_time = from_str(&time_string)?;
        Ok(Asn1GeneralizedTime {
            date_time,
            time_string_canonical: false,
            time_string,
        })
    }
    pub fn date_time(&self) -> &DateTime<Utc> {
        &self.date_time
    }
    pub fn to_date_time(&self) -> DateTime<Utc> {
        self.date_time
    }
}
impl PartialEq for Asn1GeneralizedTime {
    fn eq(&self, other: &Self) -> bool {
        self.date_time == other.date_time
    }
}
impl Hash for Asn1GeneralizedTime {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.date_time.hash(state);
    }
}
impl fmt::Display for Asn1GeneralizedTime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.time_string_canonical {
            write!(f, "{}", self.time_string)
        } else {
            write!(f, "{}", to_string_canonical(&self.date_time))
        }
    }
}
impl Asn1EncodingInternal for Asn1GeneralizedTime {
    fn get_encoding(&self, _encoding_type: crate::asn1::EncodingType) -> Box<dyn crate::asn1::asn1_encoding::Asn1Encoding> {
        todo!();
    }

    fn get_encoding_implicit(&self, _encoding_type: crate::asn1::EncodingType, _tag_class: u8, _tag_no: u8) -> Box<dyn crate::asn1::asn1_encoding::Asn1Encoding> {
        todo!();
    }
}

fn from_str(s: &str) -> Result<DateTime<Utc>> {
    if s.len() < 10 {
        return Err(BcError::with_invalid_argument("s len less 10"));
    }
    let v = s.replace(",", ".");

    if v.ends_with("Z") {
        return match v.len() {
            11 => parse_utc(&(v[0..v.len() - 1].to_string() + "00Z"), "%Y%m%d%H%M%Z"),
            13 => parse_utc(&v, "%Y%m%d%H%M%Z"),
            15 => parse_utc(&v, "%Y%m%d%H%M%S%Z"),
            17..=23 => parse_utc(&v, "%Y%m%d%H%M%S%.f%Z"),
            _ => Err(BcError::with_invalid_argument("invalid length")),
        };
    }

    match index_of_sign(&v, 10.max(v.len() - 5)) {
        None => match v.len() {
            10 => parse_local(&(v + "00"), "%Y%m%d%H%M"), // padding with minutes
            12 => parse_local(&v, "%Y%m%d%H%M"),
            14 => parse_local(&v, "%Y%m%d%H%M%S"),
            16..=22 => parse_local(&v, "%Y%m%d%H%M%S%.f"),
            _ => Err(BcError::with_invalid_argument("invalid length")),
        },
        Some(index) if index == v.len() - 5 => match v.len() {
            15 => parse_time_zone(&(v[0..v.len() - 5].to_string() + "00" + &v[(v.len() - 5)..]), "%Y%m%d%H%M%z"),
            17 => parse_time_zone(&v, "%Y%m%d%H%M%z"),
            19 => parse_time_zone(&v, "%Y%m%d%H%M%S%z"),
            21..=27 => parse_time_zone(&v, "%Y%m%d%H%M%S%.f%z"),
            _ => Err(BcError::with_invalid_argument("invalid length")),
        },
        Some(index) if index == v.len() - 3 => match v.len() {
            13 => parse_time_zone(&(v[0..v.len() - 3].to_string() + "00" + &v[(v.len() - 3)..]), "%Y%m%d%H%M%#z"),
            15 => parse_time_zone(&v, "%Y%m%d%H%M%#z"),
            17 => parse_time_zone(&v, "%Y%m%d%H%M%S%#z"),
            19..=23 => parse_time_zone(&v, "%Y%m%d%H%M%S%.f%#z"),
            _ => Err(BcError::with_invalid_argument("invalid length")),
        },
        Some(index) => Err(BcError::with_invalid_argument(format!("invalid index: {}", index))),
    }
}

fn parse_local(s: &str, fmt: &str) -> Result<DateTime<Utc>> {
    let local_date_time = NaiveDateTime::parse_from_str(s, fmt)?;
    let date_time = local_date_time.and_local_timezone(Local).unwrap();
    Ok(date_time.to_utc())
}

fn parse_utc(s: &str, fmt: &str) -> Result<DateTime<Utc>> {
    let local_date_time = NaiveDateTime::parse_from_str(s, fmt)?;
    let date_time = local_date_time.and_local_timezone(Utc).unwrap();
    Ok(date_time)
}

fn parse_time_zone(s: &str, fmt: &str) -> Result<DateTime<Utc>> {
    let local_date_time = DateTime::parse_from_str(s, fmt)?;
    let date_time = local_date_time.with_timezone(&Utc);
    Ok(date_time)
}

fn index_of_sign(s: &str, start_index: usize) -> Option<usize> {
    let result = s.chars().skip(start_index).position(|x| x == '+' || x == '-');
    if let Some(index) = result {
        return Some(index + start_index);
    }
    None
}

fn to_string_canonical(date_time: &DateTime<Utc>) -> String {
    date_time.format("%Y%m%d%H%M%S%.6fZ").to_string()
    //date_time.format("%Y%m%d%H%M%S%6f").to_string()
}

#[cfg(test)]
mod tests {
    use crate::asn1::Asn1GeneralizedTime;

    #[test]
    fn test_01() {
        let inputs = [
            "20020122122220",
            "20020122122220Z",
            "20020122122220-1000",
            "20020122122220+00",
            "20020122122220.1",
            "20020122122220.1Z",
            "20020122122220.1-1000",
            "20020122122220.1+00",
            "20020122122220.01",
            "20020122122220.01Z",
            "20020122122220.01-1000",
            "20020122122220.01+00",
            "20020122122220.001",
            "20020122122220.001Z",
            "20020122122220.001-1000",
            "20020122122220.001+00",
            "20020122122220.0001",
            "20020122122220.0001Z",
            "20020122122220.0001-1000",
            "20020122122220.0001+00",
            "20020122122220.0001+1000",
        ];

        for index in 0..inputs.len() {
            let input = inputs[index];
            assert!(Asn1GeneralizedTime::with_str(input).is_ok());
        }
    }

    #[test]
    fn encode_01() {
        //let req = hex::to_decode_with_str("180d3230323230383039313231355a").unwrap();
        //let asn1_object = asn1::Asn1Object::parse(&mut req.as_slice()).unwrap();
    }
}
