use std::collections::HashSet;
use crate::asn1::{Asn1Object, Asn1ObjectIdentifier, Asn1Sequence};
use crate::Result;
pub struct ExtendedKeyUsage {
    usage_table: HashSet<Asn1ObjectIdentifier>,
}

impl ExtendedKeyUsage {
    fn new(usage_table: HashSet<Asn1ObjectIdentifier>) -> ExtendedKeyUsage {
        ExtendedKeyUsage { usage_table }
    }
    fn from_sequence(asn1_sequence: Asn1Sequence) -> Result<Self> {
        let mut usage_table = HashSet::new();
        for element in asn1_sequence.into_iter() {
            let oid = Asn1ObjectIdentifier::from_asn1_object(element)?;
            usage_table.insert(oid);
        }
        Ok(Self::new(usage_table))
    }
    pub(crate) fn from_asn1_object(asn1_object: Asn1Object) -> Result<Self> {
        if let Asn1Object::Sequence(sequence) = asn1_object {
            return Ok(Self::from_sequence(sequence)?);
        }
        todo!()
            
    }
    pub fn iter(&self) -> std::collections::hash_set::Iter<'_, Asn1ObjectIdentifier> {
        self.usage_table.iter()
    }
    pub fn into_iter(self) -> std::collections::hash_set::IntoIter<Asn1ObjectIdentifier> {
        self.usage_table.into_iter()
    }
}

