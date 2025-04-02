// use chrono::prelude::*;
// 
use crate::Result;
// 
pub struct Asn1UtcTime {
//     date_time: DateTime<Utc>,
}
// 
impl Asn1UtcTime {
//     pub(crate) fn with_(v: &[u8]) -> Result<Asn1UtcTime> {
//         let s = String::from_utf8(v.to_vec())?;
//         Asn1UtcTime::with_str(&s)
//     }     
//     pub fn with_str(v: &str) -> Result<Asn1UtcTime> {
//         let date_time = from_str(v)?;
//         Ok(Asn1UtcTime { date_time })
//     }
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        // TODO[asn1] check for zero length
        todo!();
    }
}
// 
// fn from_str(s: &str) -> Result<DateTime<Utc>> {
//     match s.len() {
//         11 => Ok(DateTime::parse_from_str(s, "%y%m%d%H%MZ")?.to_utc()),
//         13 => Ok(DateTime::parse_from_str(s, "%y%m%d%H%M%S%z")?.to_utc()),
//         15 => Ok(DateTime::parse_from_str(s, "%y%m%d%H%M%S%.fZ")?.to_utc()),
//         17 => Ok(DateTime::parse_from_str(s, "%y%m%d%H%M%S%.f%z")?.to_utc()),
//         //_ => Err()
//     }
// }