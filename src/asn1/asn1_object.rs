use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::*;
use crate::{BcError, Result};
use std::hash::Hash;
use std::io::Read;

#[derive(Clone, Debug, Hash, PartialEq)]
pub enum Asn1Object {
    Boolean(Asn1Boolean),
    Integer(Asn1Integer),
    BitString(Asn1BitString),
    OctetString(Asn1OctetString),
    Null(Asn1Null),
    ObjectIdentifier(Asn1ObjectIdentifier),
    ObjectDescriptor(Asn1ObjectDescriptor),
    External(Asn1External),
    Enumerated(Asn1Enumerated),
    // not support EmbeddedPdv
    Utf8String(Asn1Utf8String),
    RelativeOid(Asn1RelativeOid),
    // not support Time,
    Sequence(Asn1Sequence),
    Set(Asn1Set),
    NumericString(Asn1NumericString),
    PrintableString(Asn1PrintableString),
    T61String(Asn1T61String),
    VideotexString(Asn1VideotexString),
    Ia5String(Asn1Ia5String),
    UtcTime(Asn1UtcTime),
    GeneralizedTime(Asn1GeneralizedTime),
    GraphicString(Asn1GraphicString),
    VisibleString(Asn1VisibleString),
    GeneralString(Asn1GeneralString),
    UniversalString(Asn1UniversalString),
    // not support UnrestrictedString,
    BmpString(Asn1BmpString),
    // not support Date,
    // not support TimeOfDay,
    // not support DateTime,
    // not support Duration,
    // not support ObjectIdentifierIri,
    // not support RelativeOidIri,
    Tagged(Asn1TaggedObject),
}
impl Asn1Object {
    pub fn is_boolean(&self) -> bool {
        matches!(self, Asn1Object::Boolean(_))
    }
    pub fn is_integer(&self) -> bool {
        matches!(self, Asn1Object::Integer(_))
    }
    pub fn is_bit_string(&self) -> bool {
        matches!(self, Asn1Object::BitString(_))
    }
    pub fn is_sequence(&self) -> bool {
        matches!(self, Asn1Object::Sequence(_))
    }
    pub fn is_object_identifier(&self) -> bool {
        matches!(self, Asn1Object::ObjectIdentifier(_))
    }
    pub fn is_relative_oid(&self) -> bool {
        matches!(self, Asn1Object::RelativeOid(_))
    }
    pub fn is_universal_string(&self) -> bool {
        matches!(self, Asn1Object::UniversalString(_))
    }
    pub fn is_printable_string(&self) -> bool {
        matches!(self, Asn1Object::PrintableString(_))
    }
    pub fn is_ia5_string(&self) -> bool {
        matches!(self, Asn1Object::Ia5String(_))
    }
    pub fn is_generalized_time(&self) -> bool {
        matches!(self, Asn1Object::GeneralizedTime(_))
    }
    pub fn as_boolean(&self) -> Option<&Asn1Boolean> {
        match self {
            Asn1Object::Boolean(obj) => Some(obj),
            _ => None,
        }
    }
    pub fn as_integer(&self) -> Option<&Asn1Integer> {
        match self {
            Asn1Object::Integer(obj) => Some(obj),
            _ => None,
        }
    }
    pub fn as_bit_string(&self) -> Option<&Asn1BitString> {
        match self {
            Asn1Object::BitString(obj) => Some(obj),
            _ => None,
        }
    }
    pub fn as_sequence(&self) -> Option<&Asn1Sequence> {
        match self {
            Asn1Object::Sequence(obj) => Some(obj),
            _ => None,
        }
    }
    pub fn as_tagged(&self) -> Option<&Asn1TaggedObject> {
        match self {
            Asn1Object::Tagged(obj) => Some(obj),
            _ => None,
        }
    }
    pub fn as_set(&self) -> Option<&Asn1Set> {
        match self {
            Asn1Object::Set(obj) => Some(obj),
            _ => None,
        }
    }
    pub fn as_printable_string(&self) -> Option<&Asn1PrintableString> {
        match self {
            Asn1Object::PrintableString(obj) => Some(obj),
            _ => None,
        }
    }
    pub fn as_ia5_string(&self) -> Option<&Asn1Ia5String> {
        match self {
            Asn1Object::Ia5String(obj) => Some(obj),
            _ => None,
        }
    }
    pub fn as_object_identifier(&self) -> Option<&Asn1ObjectIdentifier> {
        match self {
            Asn1Object::ObjectIdentifier(obj) => Some(obj),
            _ => None,
        }
    }
    pub fn from_read(reader: &mut dyn Read) -> Result<Self> {
        let mut asn1_reader = Asn1Read::new(reader, i32::MAX as usize);
        let result = asn1_reader.read_object()?;
        if let Some(object) = result {
            Ok(object)
        } else {
            Err(crate::BcError::with_invalid_format("No ASN.1 object found"))
        }
    }
    pub fn with_bytes(bytes: &[u8]) -> Result<Self> {
        let mut reader = bytes;
        Self::from_read(&mut reader)
    }
    pub fn as_string(&self) -> Option<&dyn Asn1String> {
        if let Asn1Object::PrintableString(obj) = self {
            Some(obj)
        } else if let Asn1Object::Ia5String(obj) = self {
            Some(obj)
        } else if let Asn1Object::Utf8String(obj) = self {
            Some(obj)
        } else {
            None
        }
    }
}
impl Asn1EncodingInternal for Asn1Object {
    fn get_encoding(&self, encoding_type: EncodingType) -> Box<dyn Asn1Encoding> {
        match self {
            Asn1Object::Boolean(obj) => obj.get_encoding(encoding_type),
            Asn1Object::Integer(obj) => obj.get_encoding(encoding_type),
            Asn1Object::BitString(obj) => obj.get_encoding(encoding_type),
            // Asn1Object::OctetString(obj) => obj.get_encoding(encoding_type),
            // Asn1Object::Null(obj) => obj.get_encoding(encoding_type),
            Asn1Object::ObjectIdentifier(obj) => obj.get_encoding(encoding_type),
            // Asn1Object::ObjectDescriptor(obj) => obj.get_encoding(encoding_type),
            // Asn1Object::External(obj) => obj.get_encoding(encoding_type),
            // Asn1Object::Enumerated(obj) => obj.get_encoding(encoding_type),
            Asn1Object::Utf8String(obj) => obj.get_encoding(encoding_type),
            Asn1Object::RelativeOid(obj) => obj.get_encoding(encoding_type),
            Asn1Object::Sequence(obj) => obj.get_encoding(encoding_type),
            Asn1Object::Set(obj) => obj.get_encoding(encoding_type),
            // Asn1Object::NumericString(obj) => obj.get_encoding(encoding_type),
            Asn1Object::PrintableString(obj) => obj.get_encoding(encoding_type),
            // Asn1Object::T61String(obj) => obj.get_encoding(encoding_type),
            // Asn1Object::VideotexString(obj) => obj.get_encoding(encoding_type),
            Asn1Object::Ia5String(obj) => obj.get_encoding(encoding_type),
            // Asn1Object::UtcTime(obj) => obj.get_encoding(encoding_type),
            // Asn1Object::GeneralizedTime(obj) => obj.get_encoding(encoding_type),
            // Asn1Object::GraphicString(obj) => obj.get_encoding(encoding_type),
            // Asn1Object::VisibleString(obj) => obj.get_encoding(encoding_type),
            // Asn1Object::GeneralString(obj) => obj.get_encoding(encoding_type),
            // Asn1Object::UniversalString(obj) => obj.get_encoding(encoding_type),
            Asn1Object::BmpString(obj) => obj.get_encoding(encoding_type),
            _ => {
                todo!("Encoding not implemented for {:?}", self);
            }
        }
    }
}
macro_rules! impl_from_for_asn1object {
    ($($t:ty => $v:ident),*) => {
        $(
            impl From<$t> for Asn1Object {
                fn from(value: $t) -> Self {
                    Asn1Object::$v(value)
                }
            }
            impl TryFrom<Asn1Object> for $t {
                type Error = BcError;
                fn try_from(value: Asn1Object) -> Result<Self> {
                    if let Asn1Object::$v(obj) = value {
                        Ok(obj)
                    } else {
                        Err(BcError::with_invalid_cast(concat!("Expected Asn1Object::", stringify!($v))))
                    }
                }
            }

        )*
    };
}

impl_from_for_asn1object! {
    Asn1Boolean => Boolean,
    Asn1Integer => Integer,
    Asn1BitString => BitString,
    Asn1OctetString => OctetString,
    Asn1Sequence => Sequence,
    Asn1ObjectIdentifier => ObjectIdentifier,
    Asn1Set => Set,
    Asn1TaggedObject => Tagged,
    Asn1RelativeOid => RelativeOid,
    Asn1Ia5String => Ia5String,
    Asn1BmpString => BmpString,
    Asn1Utf8String => Utf8String,
    Asn1GeneralizedTime => GeneralizedTime,
    Asn1PrintableString => PrintableString
}
