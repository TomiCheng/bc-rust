use chrono::{self, Datelike};
use crate::asn1;

pub struct Time {
    time_object: asn1::Asn1Object,
}

impl Time {
    fn new(time_object: asn1::Asn1Object) -> Time {
        Time {
            time_object
        }
    }

    pub fn with_local(date_time: &chrono::DateTime<chrono::Local>) -> Time {
        let utc = date_time.to_utc();
        if utc.year() < 1950 || utc.year() > 2049 {
            Self::new(super::rfc5280_asn1_utilities::create_generalized_time(utc))
        } else {
            Self::new(super::rfc5280_asn1_utilities::create_utc_time(utc))
        }
    }
}