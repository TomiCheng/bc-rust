use crate::asn1::try_from_tagged::{TryFromTagged, TryIntoTagged};
use crate::asn1::x500::AttributeTypeAndValue;
use crate::asn1::{Asn1Object, Asn1ObjectIdentifier, Asn1Set, Asn1TaggedObject};
use crate::{BcError, Result};
use std::collections::HashSet;

/// Holding class for a single Relative Distinguished Name (RDN).
///
/// ```text
/// RelativeDistinguishedName ::=
///     SET OF AttributeTypeAndValue
/// ```
pub struct Rdn {
    pub content: HashSet<AttributeTypeAndValue>,
}

impl Rdn {
    pub fn new(content: HashSet<AttributeTypeAndValue>) -> Self {
        Rdn {
            content
        }
    }
    pub fn with_element(attribute_type_and_value: AttributeTypeAndValue) -> Self {
        let mut set = HashSet::new();
        set.insert(attribute_type_and_value);
        Self::new(set)
    }
    pub fn with_elements(elements: &[AttributeTypeAndValue]) -> Self {
        let set = elements.iter().cloned().collect();
        Self::new(set)
    }
    /// Create a single valued RDN.
    pub fn with_oid_and_value(oid: Asn1ObjectIdentifier, value: Asn1Object) -> Result<Self> {
        Ok(Self::with_element(AttributeTypeAndValue::new(oid, value).into()))
    }
    pub fn with_set(set: Asn1Set) -> Result<Self> {
        let content = set.into_iter()
            .map(|attr| AttributeTypeAndValue::try_from(attr))
            .collect::<Result<HashSet<AttributeTypeAndValue>>>()?;
        Ok(Self::new(content))
    }
    
    pub fn get_content(&self) -> &HashSet<AttributeTypeAndValue> {
        &self.content
    }
}
// TODO

impl From<Rdn> for Asn1Object {
    fn from(value: Rdn) -> Self {
        value.content.into_iter()
            .map(|attr| Asn1Object::from(attr))
            .collect::<Asn1Set>()
            .into()
    }
}
impl TryFrom<Asn1Object> for Rdn {
    type Error = BcError;

    fn try_from(value: Asn1Object) -> Result<Self> {
        if let Asn1Object::Set(set) = value {
            Self::with_set(set)
        } else {
            Err(BcError::with_invalid_format("Expected Asn1Object::Sequence for Rdn"))
        }
    }
}
impl TryFromTagged for Rdn {
    fn try_from_tagged(tagged: Asn1TaggedObject, declared_explicit: bool) -> Result<Self>
    where
        Self: Sized,
    {
        let set: Asn1Set = tagged.try_into_tagged(declared_explicit)?;
        Self::with_set(set)
    }
}