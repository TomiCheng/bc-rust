use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::*;
use crate::{BcError, Result};
use std::hash::Hash;
use std::io::Read;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
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
    pub fn from_read(reader: &mut dyn Read) -> Result<Self> {
        let mut asn1_reader = Asn1Read::new(reader, i32::MAX as usize);
        let result = asn1_reader.read_object()?;
        if let Some(object) = result {
            Ok(object)
        } else {
            Err(BcError::with_invalid_format("No ASN.1 object found"))
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
            Asn1Object::OctetString(obj) => obj.get_encoding(encoding_type),
            Asn1Object::Null(obj) => obj.get_encoding(encoding_type),
            Asn1Object::ObjectIdentifier(obj) => obj.get_encoding(encoding_type),
            Asn1Object::ObjectDescriptor(obj) => obj.get_encoding(encoding_type),
            Asn1Object::External(obj) => obj.get_encoding(encoding_type),
            Asn1Object::Enumerated(obj) => obj.get_encoding(encoding_type),
            Asn1Object::Utf8String(obj) => obj.get_encoding(encoding_type),
            Asn1Object::RelativeOid(obj) => obj.get_encoding(encoding_type),
            Asn1Object::Sequence(obj) => obj.get_encoding(encoding_type),
            Asn1Object::Set(obj) => obj.get_encoding(encoding_type),
            Asn1Object::NumericString(obj) => obj.get_encoding(encoding_type),
            Asn1Object::PrintableString(obj) => obj.get_encoding(encoding_type),
            Asn1Object::T61String(obj) => obj.get_encoding(encoding_type),
            Asn1Object::VideotexString(obj) => obj.get_encoding(encoding_type),
            Asn1Object::Ia5String(obj) => obj.get_encoding(encoding_type),
            Asn1Object::UtcTime(obj) => obj.get_encoding(encoding_type),
            Asn1Object::GeneralizedTime(obj) => obj.get_encoding(encoding_type),
            Asn1Object::GraphicString(obj) => obj.get_encoding(encoding_type),
            Asn1Object::VisibleString(obj) => obj.get_encoding(encoding_type),
            Asn1Object::GeneralString(obj) => obj.get_encoding(encoding_type),
            Asn1Object::UniversalString(obj) => obj.get_encoding(encoding_type),
            Asn1Object::BmpString(obj) => obj.get_encoding(encoding_type),
            Asn1Object::Tagged(obj) => obj.get_encoding(encoding_type),
        }
    }

    fn get_encoding_implicit(
        &self,
        encoding_type: EncodingType,
        tag_class: u8,
        tag_no: u8,
    ) -> Box<dyn Asn1Encoding> {
        match self {
            Asn1Object::Boolean(obj) => obj.get_encoding_implicit(encoding_type, tag_class, tag_no),
            Asn1Object::Integer(obj) => obj.get_encoding_implicit(encoding_type, tag_class, tag_no),
            Asn1Object::BitString(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::OctetString(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::Null(obj) => obj.get_encoding_implicit(encoding_type, tag_class, tag_no),
            Asn1Object::ObjectIdentifier(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::ObjectDescriptor(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::External(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::Enumerated(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::Utf8String(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::RelativeOid(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::Sequence(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::Set(obj) => obj.get_encoding_implicit(encoding_type, tag_class, tag_no),
            Asn1Object::NumericString(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::PrintableString(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::T61String(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::VideotexString(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::Ia5String(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::UtcTime(obj) => obj.get_encoding_implicit(encoding_type, tag_class, tag_no),
            Asn1Object::GeneralizedTime(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::GraphicString(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::VisibleString(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::GeneralString(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::UniversalString(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::BmpString(obj) => {
                obj.get_encoding_implicit(encoding_type, tag_class, tag_no)
            }
            Asn1Object::Tagged(obj) => obj.get_encoding_implicit(encoding_type, tag_class, tag_no),
        }
    }
}
macro_rules! impl_from_for_asn1object {
    ($($t:ty => $v:ident => $t1:ident),*) => {
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
            impl Asn1Object {
                pub fn $t1(&self) -> bool {
                    matches!(self, Asn1Object::$v(_))
                }
            }
        )*
    };
}

impl_from_for_asn1object! {
    Asn1Boolean => Boolean => is_boolean,
    Asn1Integer => Integer => is_integer,
    Asn1BitString => BitString => is_bit_string,
    Asn1OctetString => OctetString => is_octet_string,
    Asn1Null => Null => is_null,
    Asn1ObjectIdentifier => ObjectIdentifier => is_object_identifier,
    Asn1ObjectDescriptor => ObjectDescriptor => is_object_descriptor,
    Asn1External => External => is_external,
    Asn1Enumerated => Enumerated => is_enumerated,
    Asn1Utf8String => Utf8String => is_utf8_string,
    Asn1RelativeOid => RelativeOid => is_relative_oid,
    Asn1Sequence => Sequence => is_sequence,
    Asn1Set => Set => is_set,
    Asn1NumericString => NumericString => is_numeric_string,
    Asn1PrintableString => PrintableString => is_printable_string,
    Asn1T61String => T61String => is_t61_string,
    Asn1VideotexString => VideotexString => is_videotex_string,
    Asn1Ia5String => Ia5String => is_ia5_string,
    Asn1UtcTime => UtcTime => is_utc_time,
    Asn1GeneralizedTime => GeneralizedTime => is_generalized_time,
    Asn1GraphicString => GraphicString => is_graphic_string,
    Asn1VisibleString => VisibleString => is_visible_string,
    Asn1GeneralString => GeneralString => is_general_string,
    Asn1UniversalString => UniversalString => is_universal_string,
    Asn1BmpString => BmpString => is_bmp_string,
    Asn1TaggedObject => Tagged => is_tagged_object
}
