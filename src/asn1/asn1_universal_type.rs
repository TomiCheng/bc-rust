use crate::asn1::{asn1_tags, Asn1Integer, Asn1Object, Asn1Sequence, Asn1TaggedObject};
use crate::Result;
pub(crate) trait Asn1UniversalType<TAsn1Type> {
    fn checked_cast(&self, asn1_object: Asn1Object) -> Result<TAsn1Type>;
    fn implicit_constructed(&self, sequence: Asn1Sequence) -> Result<TAsn1Type>;
    fn get_tagged(&self, tagged_object: Asn1TaggedObject, declared_explicit: bool) -> Result<TAsn1Type>
    where Self: Sized {
        let result = tagged_object.get_base_universal(declared_explicit, self)?;
        Ok(result)
    }
}

