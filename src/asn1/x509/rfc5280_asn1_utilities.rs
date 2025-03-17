use chrono;

use crate::asn1;
use crate::util::date::date_time_utilities;

pub(crate) fn create_generalized_time(
    date_time: &chrono::DateTime<chrono::Utc>,
) -> asn1::Asn1Object {
    let v = date_time_utilities::with_precision_second(date_time);
    //asn1::Asn1Object::with_der_generalized_time(DerGerneralizedTimeImpl::with_utc)
    todo!()
}

pub(crate) fn create_utc_time(date_time: chrono::DateTime<chrono::Utc>) -> asn1::Asn1Object {
    //asn1::Asn1Object::with_der_utc_time(DerUtcTimeImpl::with_utc(date_time, 2049))
    todo!()
}
