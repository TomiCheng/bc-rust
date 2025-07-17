use std::hash::Hash;
use crate::asn1::{Asn1Object, Asn1ObjectIdentifier, Asn1Sequence, Asn1TaggedObject};
use crate::{BcError, Result};
use crate::asn1::try_from_tagged::{TryFromTagged, TryIntoTagged};

/// Holding class for the AttributeTypeAndValue structures that make up an RDN.
///
/// ```text
/// AttributeTypeAndValue ::= SEQUENCE {
///     type         OBJECT IDENTIFIER,
///     value        ANY DEFINED BY type
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Hash, Eq)]
pub struct AttributeTypeAndValue {
    pub attribute_type: Asn1ObjectIdentifier,
    pub value: Asn1Object,
}

impl AttributeTypeAndValue {
    pub fn new(attribute_type: Asn1ObjectIdentifier, value: Asn1Object) -> Self {
        AttributeTypeAndValue {
            attribute_type,
            value
        }
    }
    pub fn with_sequence(sequence: Asn1Sequence) -> Result<Self> {
        if sequence.len() != 2 {
            return Err(BcError::with_invalid_format("Invalid sequence length for AttributeTypeAndValue"));
        }
        let mut iter = sequence.into_iter();
        let attribute_type = iter.next()
            .ok_or_else(|| BcError::with_invalid_format("Missing attribute type in AttributeTypeAndValue"))?
            .try_into()?;
        let value = iter.next()
            .ok_or_else(|| BcError::with_invalid_format("Missing value in AttributeTypeAndValue"))?;

        Ok(Self::new(attribute_type, value))
    }
    pub fn get_attribute_type(&self) -> &Asn1ObjectIdentifier {
        &self.attribute_type
    }
    pub fn get_value(&self) -> &Asn1Object {
        &self.value
    }
}
impl From<AttributeTypeAndValue> for Asn1Object {
    fn from(value: AttributeTypeAndValue) -> Self {
        Asn1Object::Sequence(Asn1Sequence::new(vec![
            value.attribute_type.into(),
            value.value,
        ]))
    }
}
impl TryFrom<Asn1Object> for AttributeTypeAndValue {
    type Error = BcError;

    fn try_from(value: Asn1Object) -> Result<Self> {
        if let Asn1Object::Sequence(seq) = value {
            Self::with_sequence(seq)
        } else {
            Err(BcError::with_invalid_format("Expected Asn1Object::Sequence for AttributeTypeAndValue"))
        }
    }
}
impl TryFromTagged for AttributeTypeAndValue {
    fn try_from_tagged(tagged: Asn1TaggedObject, declared_explicit: bool) -> Result<Self>
    where
        Self: Sized,
    {
        let sequence: Asn1Sequence = tagged.try_into_tagged(declared_explicit)?;
        Self::with_sequence(sequence)
    }
}