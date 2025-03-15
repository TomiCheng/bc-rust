use chrono;

use crate::asn1;

pub(crate) fn create_generalized_time(
    date_time: chrono::DateTime<chrono::Utc>,
) -> asn1::Asn1Object {
    //asn1::Asn1Object::with_der_generalized_time(DerGerneralizedTimeImpl::with_p)
    todo!()
}

pub(crate) fn create_utc_time(date_time: chrono::DateTime<chrono::Utc>) -> asn1::Asn1Object {
    //asn1::Asn1Object::with_der_utc_time(DerUtcTimeImpl::with_utc(date_time, 2049))
    todo!()
}
