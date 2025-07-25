use crate::asn1::x509::Time;
use crate::asn1::{Asn1Object, Asn1Sequence};
use crate::{BcError, Result};
pub struct Validity {
    not_before: Time,
    not_after: Time,
}

impl Validity {
    pub fn new(not_before: Time, not_after: Time) -> Self {
        Validity { not_before, not_after }
    }
    fn from_sequence(sequence: Asn1Sequence) -> Result<Self> {
        if sequence.len() != 2 {
            return Err(crate::BcError::with_invalid_format(format!("bad sequence size: {}", sequence.len())));
        }
        let mut iter = sequence.into_iter();
        let not_before = Time::from_asn1_object(iter.next().unwrap())?;
        let not_after = Time::from_asn1_object(iter.next().unwrap())?;
        Ok(Validity::new(not_before, not_after))
    }
    pub fn not_before(&self) -> &Time {
        &self.not_before
    }
    pub fn not_after(&self) -> &Time {
        &self.not_after
    }
}
impl TryFrom<Asn1Object> for Validity {
    type Error = BcError;

    fn try_from(value: Asn1Object) -> Result<Self> {
        if let Ok(sequence) = value.try_into() {
            return Validity::from_sequence(sequence);
        }
        todo!()
    }
}
