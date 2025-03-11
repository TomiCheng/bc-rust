use std::fmt::{Display, Formatter};
use std::io::{Read, Write};

use super::asn1_encoding::Asn1Encoding;
use super::Asn1Encodable;
use super::Asn1Read;
use super::Asn1Write;
use super::DerBitStringImpl;
use super::DerBooleanImpl;
use super::DerIntegerImpl;
use super::DerNullImpl;
use super::DerObjectIdentifierImpl;
use super::DerOctetStringImpl;
use super::DerSequenceImpl;
use crate::Result;

pub(crate) trait Asn1ObjectImpl: Asn1Encodable + Display {}

#[derive(Clone, Debug)]
pub enum Asn1Object {
    DerBoolean(DerBooleanImpl),
    DerInteger(DerIntegerImpl),
    DerBitString(DerBitStringImpl),
    DerOctetString(DerOctetStringImpl),
    DerNull(DerNullImpl),
    DerObjectIdentifier(DerObjectIdentifierImpl),
    DerSequence(DerSequenceImpl),
}

macro_rules! is_variant {
    ($name:ident, $variant:pat) => {
        pub fn $name(&self) -> bool {
            match self {
                $variant => true,
                _ => false,
            }
        }
    };
}

macro_rules! as_variant {
    ($name:ident, $variant:path, $type:ty) => {
        pub fn $name(&self) -> &$type {
            match self {
                $variant(impl_) => impl_,
                _ => panic!("Not a match"),
            }
        }
    };
}

impl Asn1Object {
    pub fn new_der_boolean(value: DerBooleanImpl) -> Self {
        Asn1Object::DerBoolean(value)
    }
    pub fn new_der_integer(value: DerIntegerImpl) -> Self {
        Asn1Object::DerInteger(value)
    }
    pub fn new_der_bit_string(value: DerBitStringImpl) -> Self {
        Asn1Object::DerBitString(value)
    }
    pub fn new_der_null(value: DerNullImpl) -> Self {
        Asn1Object::DerNull(value)
    }

    pub fn with_null() -> Self {
        Asn1Object::DerNull(DerNullImpl::new())
    }

    pub fn with_bool(value: bool) -> Self {
        Asn1Object::DerBoolean(DerBooleanImpl::new(value))
    }
    pub fn with_i32(value: i32) -> Self {
        Asn1Object::DerInteger(DerIntegerImpl::with_i32(value))
    }
    fn get_impl(&self) -> &dyn Asn1ObjectImpl {
        match self {
            Asn1Object::DerBoolean(der_boolean) => der_boolean,
            Asn1Object::DerInteger(der_integer) => der_integer,
            Asn1Object::DerBitString(der_bit_string) => der_bit_string,
            Asn1Object::DerOctetString(der_octet_string) => der_octet_string,
            Asn1Object::DerNull(der_null) => der_null,
            Asn1Object::DerObjectIdentifier(der_object_identifier) => der_object_identifier,
            Asn1Object::DerSequence(der_sequence) => der_sequence,
        }
    }

    is_variant!(is_der_boolean, Asn1Object::DerBoolean(_));
    is_variant!(is_der_integer, Asn1Object::DerInteger(_));
    is_variant!(is_der_bit_string, Asn1Object::DerBitString(_));
    is_variant!(is_der_octet_string, Asn1Object::DerOctetString(_));
    is_variant!(is_der_null, Asn1Object::DerNull(_));
    is_variant!(is_der_sequence, Asn1Object::DerSequence(_));

    as_variant!(as_der_boolean, Asn1Object::DerBoolean, DerBooleanImpl);
    as_variant!(as_der_integer, Asn1Object::DerInteger, DerIntegerImpl);
    as_variant!(
        as_der_bit_string,
        Asn1Object::DerBitString,
        DerBitStringImpl
    );
    as_variant!(
        as_der_octet_string,
        Asn1Object::DerOctetString,
        DerOctetStringImpl
    );
    as_variant!(as_der_null, Asn1Object::DerNull, DerNullImpl);

    /// Read a base ASN.1 object from a Read.
    /// # Arguments
    /// * `reader` - The Read to parse.
    /// # Returns
    /// The base ASN.1 object represented by the byte array.
    /// # Errors
    /// If there is a problem parsing the data.
    pub fn parse(reader: &mut dyn Read) -> Result<Asn1Object> {
        let mut asn1_reader = Asn1Read::new(reader, i32::MAX as usize);
        asn1_reader.read_object()
    }
}

impl Display for Asn1Object {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        self.get_impl().fmt(f)
    }
}

impl Asn1Encodable for Asn1Object {
    fn get_encoded_with_encoding(&self, encoding: &str) -> Result<Vec<u8>> {
        self.get_impl().get_encoded_with_encoding(encoding)
    }

    fn encode_to_with_encoding(&self, writer: &mut dyn Write, encoding: &str) -> Result<usize> {
        self.get_impl().encode_to_with_encoding(writer, encoding)
    }
}

impl PartialEq for Asn1Object {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Asn1Object::DerBoolean(der_boolean) => {
                if let Asn1Object::DerBoolean(other_der_boolean) = other {
                    return der_boolean == other_der_boolean;
                }
                false
            }
            Asn1Object::DerObjectIdentifier(der_object_identifier) => {
                if let Asn1Object::DerObjectIdentifier(other_der_object_identifier) = other {
                    return der_object_identifier == other_der_object_identifier;
                }
                false
            }
            _ => false,
        }
    }
}

impl From<DerBooleanImpl> for Asn1Object {
    fn from(value: DerBooleanImpl) -> Self {
        Asn1Object::DerBoolean(value)
    }
}

pub(crate) fn get_encoded_with_encoding(
    encoding_str: &str,
    encoding: &dyn Asn1Encoding,
) -> Result<Vec<u8>> {
    let length = encoding.get_length();
    let mut result = Vec::with_capacity(length);
    let mut asn1_writer = Asn1Write::create_with_encoding(&mut result, encoding_str);
    encoding.encode(&mut asn1_writer)?;
    Ok(result)
}

pub(crate) fn encode_to_with_encoding(
    writer: &mut dyn Write,
    encoding_str: &str,
    asn1_encoding: &dyn Asn1Encoding,
) -> Result<usize> {
    let mut asn1_writer = Asn1Write::create_with_encoding(writer, encoding_str);
    let mut result = 0;
    result += asn1_encoding.encode(&mut asn1_writer)?;
    Ok(result)
}
