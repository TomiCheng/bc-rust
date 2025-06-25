use crate::asn1::{Asn1BitString, Asn1ObjectIdentifier, Asn1OctetString, Asn1Boolean, Asn1Integer, Asn1Null, Asn1ObjectDescriptor, Asn1Enumerated, Asn1External, Asn1Utf8String, Asn1Set, EncodingType};
use crate::asn1::asn1_encoding::Asn1Encoding;

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
    Utf8String(Asn1Utf8String),
    Set(Asn1Set), 
}

impl Asn1Object {
    pub fn is_boolean(&self) -> bool {
        matches!(self, Asn1Object::Boolean(_))
    }
    pub fn as_boolean(&self) -> Option<&Asn1Boolean> {
        match self {
            Asn1Object::Boolean(obj) => Some(obj),
            _ => None,
        }
    }
}

impl Asn1Object {
    pub(crate) fn get_encoding(&self, encoding_type: EncodingType) -> Box<dyn Asn1Encoding> {
        match self {
            Asn1Object::Boolean(obj) => obj.get_encoding(encoding_type),
            //Asn1Object::Integer(obj) => Box::new(obj.get_encoding(encoding_type)),
            //Asn1Object::BitString(obj) => Box::new(obj.get_encoding(encoding_type)),
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
}