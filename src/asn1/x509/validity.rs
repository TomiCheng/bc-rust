use anyhow::ensure;
use crate::asn1::Asn1Sequence;
use crate::asn1::x509::Time;
use crate::{Result, Error};

pub struct Validity {
    not_before: Time,
    not_after: Time,
}

impl Validity {
    pub fn new(not_before: Time, not_after: Time) -> Self {
        Validity { not_before, not_after }
    }
    
    pub fn with_asn1_sequence(seq: &Asn1Sequence) -> Result<Self> {
        ensure!(seq.len() == 2, Error::invalid_argument("Validity sequence must have exactly 2 elements", "seq"));
        
        todo!();
    }

    pub fn not_before(&self) -> &Time {
        &self.not_before
    }

    pub fn not_after(&self) -> &Time {
        &self.not_after
    }
}