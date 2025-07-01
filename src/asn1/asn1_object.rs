use std::io::{Read, Write};
use crate::asn1::{Asn1BitString, Asn1ObjectIdentifier, Asn1OctetString, Asn1Boolean, Asn1Integer};
use crate::asn1::{Asn1Null, Asn1ObjectDescriptor, Asn1Enumerated, Asn1External, Asn1Utf8String,};
use crate::asn1::{Asn1Set, EncodingType, Asn1Read, Asn1Encodable, Asn1Write, Asn1TaggedObject,};
use crate::asn1::{Asn1Sequence, Asn1PrintableString, Asn1Ia5String, Asn1UtcTime, Asn1GeneralizedTime};
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::{BcError, Result};
use crate::asn1::asn1_relative_oid::Asn1RelativeOid;

#[derive(Clone, Debug)]
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
    // not support EMBEDDED_PDV
    Utf8String(Asn1Utf8String),
    RelativeOid(Asn1RelativeOid),
    Time,
    Sequence(Asn1Sequence),
    Set(Asn1Set),
    NumericString,
    PrintableString(Asn1PrintableString),
    T61String,
    VideotexString,
    Ia5String(Asn1Ia5String),
    UtcTime(Asn1UtcTime),
    GeneralizedTime(Asn1GeneralizedTime),
    GraphicString,
    VisibleString,
    GeneralString,
    UniversalString,
    UnrestrictedString,
    BmpString,
    Date,
    TimeOfDay,
    DateTime,
    Duration,
    ObjectIdentifierIri,
    RelativeOidIri,
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
    pub fn as_object_identifier(&self) -> Option<&Asn1ObjectIdentifier> {
        match self {
            Asn1Object::ObjectIdentifier(obj) => Some(obj),
            _ => None,
        }
    }
    pub(crate) fn get_encoding(&self, encoding_type: EncodingType) -> Box<dyn Asn1Encoding> {
        match self {
            Asn1Object::Boolean(obj) => obj.get_encoding(encoding_type),
            Asn1Object::Integer(obj) => obj.get_encoding(encoding_type),
            Asn1Object::BitString(obj) => obj.get_encoding(encoding_type),
            //Asn1Object::OctetString(obj) => Box::new(obj.get_encoding(encoding_type)),
            //Asn1Object::Null(obj) => Box::new(obj.get_encoding(encoding_type)),
            //Asn1Object::ObjectIdentifier(obj) => Box::new(obj.get_encoding(encoding_type)),
            //Asn1Object::ObjectDescriptor(obj) => Box::new(obj.get_encoding(encoding_type)),
            //Asn1Object::External(obj) => Box::new(obj.get_encoding(encoding_type)),
            //Asn1Object::Enumerated(obj) => Box::new(obj.get_encoding(encoding_type)),
            //Asn1Object::Utf8String(obj) => Box::new(obj.get_encoding(encoding_type)),
            //Asn1Object::Set(obj) => Box::new(obj.get_encoding(encoding_type)),
            _ =>  // Placeholder for other object types
            todo!()
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
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut reader = bytes;
        Self::from_read(&mut reader)
    }
}

impl Asn1Encodable for Asn1Object {
    fn encode_to(&self, writer: &mut dyn Write, encoding_type: EncodingType) -> Result<usize> {
        let mut asn1_writer = Asn1Write::new(writer, encoding_type);
        let asn1_encoding = self.get_encoding(encoding_type);
        asn1_encoding.encode(&mut asn1_writer)
    }
}

impl From<Asn1Boolean> for Asn1Object {
    fn from(value: Asn1Boolean) -> Self {
        Asn1Object::Boolean(value)
    }
}
impl From<Asn1BitString> for Asn1Object {
    fn from(value: Asn1BitString) -> Self {
        Asn1Object::BitString(value)
    }
}
impl From<Asn1OctetString> for Asn1Object {
    fn from(value: Asn1OctetString) -> Self {
        Asn1Object::OctetString(value)
    }
}
impl From<Asn1Sequence> for Asn1Object {
    fn from(value: Asn1Sequence) -> Self {
        Asn1Object::Sequence(value)
    }
}
impl From<Asn1Set> for Asn1Object {
    fn from(value: Asn1Set) -> Self {
        Asn1Object::Set(value)
    }
}
impl From<Asn1TaggedObject> for Asn1Object {
    fn from(value: Asn1TaggedObject) -> Self {
        Asn1Object::Tagged(value)
    }
}
impl From<Asn1RelativeOid> for Asn1Object {
    fn from(value: Asn1RelativeOid) -> Self {
        Asn1Object::RelativeOid(value)
    }
}

impl TryFrom<Asn1Object> for Asn1RelativeOid {
    type Error = BcError;

    fn try_from(value: Asn1Object) -> Result<Self> {
        if let Asn1Object::RelativeOid(oid) = value {
            Ok(oid)
        } else {
            Err(BcError::with_invalid_cast("Expected Asn1Object::RelativeOid"))
        }
    }
}