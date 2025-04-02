use anyhow::{bail, ensure};
// use std::fmt;
// use std::io;
// use std::sync;
// use std::any;
// 
use chrono::prelude::*;
 
// use super::*;
use crate::{Error, Result};
// 
// #[derive(Debug, Clone)]
/// GeneralizedTime ASN.1 type
pub struct Asn1GeneralizedTime {
    date_time: DateTime<Utc>,
    time_string_canonical: bool,
    time_string: String,
}
// 
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
// 
//     pub(crate) fn get_contents(&self, encoding_type: asn1_write::EncodingType) -> Vec<u8> {
//         if encoding_type == asn1_write::EncodingType::Der && self.time_string_canonical {
//             return to_string_canonical(&self.date_time).as_bytes().to_vec();
//         }
//         self.time_string.as_bytes().to_vec()
//     }
// 
//     fn get_encoding_with_type(
//         &self,
//         encode_type: asn1_write::EncodingType,
//     ) -> Box<dyn asn1_encoding::Asn1Encoding> {
//         Box::new(primitive_encoding::PrimitiveEncoding::new(
//             asn1_tags::UNIVERSAL,
//             asn1_tags::INTEGER,
//             sync::Arc::new(self.get_contents(encode_type)),
//         ))
//     }
// }
// 
// impl Asn1Object for Asn1GeneralizedTime {}
// impl Asn1Encodable for Asn1GeneralizedTime {
//     fn get_encoded_with_encoding(&self, encoding_str: &str) -> Result<Vec<u8>> {
//         let encoding = self.get_encoding_with_type(asn1_write::get_encoding_type(encoding_str));
//         asn1_object::get_encoded_with_encoding(encoding_str, encoding.as_ref())
//     }
// 
//     fn encode_to_with_encoding(
//         &self,
//         writer: &mut dyn io::Write,
//         encoding_str: &str,
//     ) -> Result<usize> {
//         let asn1_encoding =
//             self.get_encoding_with_type(asn1_write::get_encoding_type(encoding_str));
//         asn1_object::encode_to_with_encoding(writer, encoding_str, asn1_encoding.as_ref())
//     }
// }
// impl Asn1Convertiable for Asn1GeneralizedTime {
//     fn to_asn1_object(&self) -> sync::Arc<dyn Asn1Object> {
//         sync::Arc::new(self.clone())
//     }
//     fn as_any(&self) -> sync::Arc<dyn any::Any> {
//         sync::Arc::new(self.clone())
//     }
// }
// impl fmt::Display for Asn1GeneralizedTime {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         write!(f, "{}", self.date_time)
//     }
}


fn from_str(s: &str) -> Result<DateTime<Utc>> {
    ensure!(
        s.len() >= 10,
        Error::invalid_argument("s len less 10", "s")
    );

    let v = s.replace(",", ".");

    if v.ends_with("Z") {
        return match v.len() {
            11 => parse_utc(&(v[0..v.len() - 1].to_string() + "00Z"), "%Y%m%d%H%M%Z"),
            13 => parse_utc(&v, "%Y%m%d%H%M%Z"),
            15 => parse_utc(&v, "%Y%m%d%H%M%S%Z"),
            17..=23 => parse_utc(&v, "%Y%m%d%H%M%S%.f%Z"),
            _ => bail!(Error::invalid_argument("invalid length", "s")),
        };
    }

    match index_of_sign(&v, 10.max(v.len() - 5)) {
        None => match v.len() {
            10 => parse_local(&(v + "00"), "%Y%m%d%H%M"), // padding with minutes
            12 => parse_local(&v, "%Y%m%d%H%M"),
            14 => parse_local(&v, "%Y%m%d%H%M%S"),
            16..=22 => parse_local(&v, "%Y%m%d%H%M%S%.f"),
            _ => bail!(Error::invalid_argument("invalid length", "s")),
        },
        Some(index) if index == v.len() - 5 => match v.len() {
            15 => parse_time_zone(
                &(v[0..v.len() - 5].to_string() + "00" + &v[(v.len() - 5)..]),
                "%Y%m%d%H%M%z",
            ),
            17 => parse_time_zone(&v, "%Y%m%d%H%M%z"),
            19 => parse_time_zone(&v, "%Y%m%d%H%M%S%z"),
            21..=27 => parse_time_zone(&v, "%Y%m%d%H%M%S%.f%z"),
            _ => bail!(Error::invalid_argument("invalid length", "s")),
        },
        Some(index) if index == v.len() - 3 => match v.len() {
            13 => parse_time_zone(
                &(v[0..v.len() - 3].to_string() + "00" + &v[(v.len() - 3)..]),
                "%Y%m%d%H%M%#z",
            ),
            15 => parse_time_zone(&v, "%Y%m%d%H%M%#z"),
            17 => parse_time_zone(&v, "%Y%m%d%H%M%S%#z"),
            19..=23 => parse_time_zone(&v, "%Y%m%d%H%M%S%.f%#z"),
            _ => bail!(Error::invalid_argument("invalid length", "s")),
        },
        Some(index) => bail!(Error::invalid_argument(
            &format!("invalid index: {index}"),
            "index"
        )),
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
    let result = s
        .chars()
        .skip(start_index)
        .position(|x| x == '+' || x == '-');
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
            "20020122122220.0001+1000"
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