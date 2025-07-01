use crate::asn1::Asn1Object;
use crate::asn1::x509::Time;
use crate::Result;
pub struct Validity {
    not_before: Time,
    not_after: Time,
}

impl Validity {
    pub fn new(not_before: Time, not_after: Time) -> Self {
        Validity {
            not_before,
            not_after,
        }
    }
    pub(crate) fn from_asn1_object(p0: &Asn1Object) -> Result<Self> {
        todo!()
    }
    pub fn not_before(&self) -> &Time {
        &self.not_before
    }
    pub fn not_after(&self) -> &Time {
        &self.not_after
    }
}