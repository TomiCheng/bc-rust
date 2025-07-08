use crate::asn1::{Asn1Object, Asn1OctetString, Asn1Sequence};
use crate::Result;
pub(crate) trait Asn1UniversalType<TAsn1Type> where Self: Sized {
    fn checked_cast(&self, asn1_object: Asn1Object) -> Result<TAsn1Type>;
    fn implicit_constructed(&self, _: Asn1Sequence) -> Result<TAsn1Type> {
        Err(crate::BcError::with_invalid_operation("unexpected implicit constructed encoding", ))
    }
    fn implicit_primitive(&self, _: Asn1OctetString) -> Result<TAsn1Type> {
        Err(crate::BcError::with_invalid_operation("unexpected implicit primitive encoding", ))
    }
}

