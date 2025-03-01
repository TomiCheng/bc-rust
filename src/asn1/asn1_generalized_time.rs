use super::asn1_write::EncodingType;
use crate::error::BcError;
use crate::Result;
use chrono::{DateTime, Utc};

/// GeneralizedTime ASN.1 type
pub struct Asn1GeneralizedTime {
    date_time: DateTime<Utc>,
    time_string_canonical: bool,
    time_string: String,
}

impl Asn1GeneralizedTime {
    pub fn with_string(time_string: &str) -> Result<Self> {
        let date_time = from_string(time_string)?;
        Ok(Asn1GeneralizedTime {
            date_time,
            time_string_canonical: false,
            time_string: time_string.to_string(),
        })
    }
    pub fn with_date_time(date_time: &DateTime<Utc>) -> Self {
        Asn1GeneralizedTime {
            date_time: date_time.clone(),
            time_string_canonical: true,
            time_string: to_string_canonical(date_time),
        }
    }

    pub fn get_time_string(&self) -> &str {
        &self.time_string
    }
    pub fn to_date_time(&self) -> DateTime<Utc> {
        self.date_time
    }

    pub(crate) fn get_contents(&self, encoding_type: &EncodingType) -> Vec<u8> {
        if encoding_type == &EncodingType::Der && self.time_string_canonical {
            return to_string_canonical(&self.date_time).as_bytes().to_vec();
        }
        self.time_string.as_bytes().to_vec()
    }
}

fn from_string(s: &str) -> Result<DateTime<Utc>> {
    if s.len() < 10 {
        return Err(BcError::InvalidFormat("".to_string()));
    }

    let v = s.replace(",", ".");

    if v.ends_with("Z") {
        return match v.len() {
            11 => parse_utc(&v, "%Y%m%d%H%Z"),
            13 => parse_utc(&v, "%Y%m%d%H%M%Z"),
            15 => parse_utc(&v, "%Y%m%d%H%M%S%Z"),
            17 => parse_utc(&v, "%Y%m%d%H%M%S%.f%Z"),
            18 => parse_utc(&v, "%Y%m%d%H%M%S%.f%Z"),
            19 => parse_utc(&v, "%Y%m%d%H%M%S%.f%Z"),
            20 => parse_utc(&v, "%Y%m%d%H%M%S%.f%Z"),
            21 => parse_utc(&v, "%Y%m%d%H%M%S%.f%Z"),
            22 => parse_utc(&v, "%Y%m%d%H%M%S%.f%Z"),
            23 => parse_utc(&v, "%Y%m%d%H%M%S%.f%Z"),
            _ => Err(BcError::InvalidFormat("".to_string())),
        };
    }

    return match index_of_sign(&v, 10.max(v.len() - 5)) {
        None => match v.len() {
            10 => parse_utc(&v, "%Y%m%d%H"),
            11 => parse_utc(&v, "%Y%m%d%H%M"),
            12 => parse_utc(&v, "%Y%m%d%H%M%S"),
            13 => parse_utc(&v, "%Y%m%d%H%M%S%.f"),
            14 => parse_utc(&v, "%Y%m%d%H%M%S%.f"),
            15 => parse_utc(&v, "%Y%m%d%H%M%S%.f"),
            16 => parse_utc(&v, "%Y%m%d%H%M%S%.f"),
            17 => parse_utc(&v, "%Y%m%d%H%M%S%.f"),
            18 => parse_utc(&v, "%Y%m%d%H%M%S%.f"),
            19 => parse_utc(&v, "%Y%m%d%H%M%S%.f"),
            20 => parse_utc(&v, "%Y%m%d%H%M%S%.f"),
            21 => parse_utc(&v, "%Y%m%d%H%M%S%.f"),
            22 => parse_utc(&v, "%Y%m%d%H%M%S%.f"),
            _ => Err(BcError::InvalidFormat("".to_string())),
        },
        Some(index) if index == v.len() - 5 => match v.len() {
            15 => parse_utc(&v, "%Y%m%d%H%z"),
            17 => parse_utc(&v, "%Y%m%d%H%M%z"),
            19 => parse_utc(&v, "%Y%m%d%H%M%S%z"),
            21 => parse_utc(&v, "%Y%m%d%H%M%S%.f%z"),
            22 => parse_utc(&v, "%Y%m%d%H%M%S%.f%z"),
            23 => parse_utc(&v, "%Y%m%d%H%M%S%.f%z"),
            24 => parse_utc(&v, "%Y%m%d%H%M%S%.f%z"),
            25 => parse_utc(&v, "%Y%m%d%H%M%S%.f%z"),
            26 => parse_utc(&v, "%Y%m%d%H%M%S%.f%z"),
            27 => parse_utc(&v, "%Y%m%d%H%M%S%.f%z"),
            _ => Err(BcError::InvalidFormat("".to_string())),
        },
        Some(index) if index == v.len() - 3 => match v.len() {
            13 => parse_utc(&v, "%Y%m%d%H%:::z"),
            15 => parse_utc(&v, "%Y%m%d%H%M%:::z"),
            17 => parse_utc(&v, "%Y%m%d%H%M%S%:::z"),
            19 => parse_utc(&v, "%Y%m%d%H%M%S%.f%:::z"),
            20 => parse_utc(&v, "%Y%m%d%H%M%S%.f%:::z"),
            21 => parse_utc(&v, "%Y%m%d%H%M%S%.f%:::z"),
            22 => parse_utc(&v, "%Y%m%d%H%M%S%.f%:::z"),
            23 => parse_utc(&v, "%Y%m%d%H%M%S%.f%:::z"),
            24 => parse_utc(&v, "%Y%m%d%H%M%S%.f%:::z"),
            25 => parse_utc(&v, "%Y%m%d%H%M%S%.f%:::z"),
            _ => Err(BcError::InvalidFormat("".to_string())),
        },
        _ => Err(BcError::InvalidFormat("".to_string())),
    };
}

fn parse_utc(s: &str, fmt: &str) -> Result<DateTime<Utc>> {
    Ok(DateTime::parse_from_str(s, fmt)
        .map(|d| d.to_utc())
        .map_err(|e| BcError::ParseError {
            msg: "".to_string(),
            source: e,
        })?)
}

fn index_of_sign(s: &str, start_index: usize) -> Option<usize> {
    s.chars()
        .skip(start_index)
        .position(|x| x == '+' || x == '-')
}

fn to_string_canonical(date_time: &DateTime<Utc>) -> String {
    date_time.format("%Y%m%d%H%M%S%6f%Z").to_string()
}
