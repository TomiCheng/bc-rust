use super::asn1_encoding::Asn1Encoding;
use super::asn1_object::Asn1ObjectInternal;
use super::asn1_tags::{BOOLEAN, UNIVERSAL};
use super::asn1_write::EncodingType;
use super::primitive_encoding::PrimitiveEncoding;
use super::{Asn1Convertiable, Asn1Encodable, Asn1ObjectImpl};
use std::fmt::{Display, Formatter};
use std::rc::Rc;

pub struct DerBoolean {
    value: u8,
}

impl DerBoolean {
    pub fn new(value: bool) -> Self {
        DerBoolean {
            value: if value { 0xFF } else { 0x00 },
        }
    }

    pub fn with_i32(value: i32) -> Self {
        DerBoolean::new(value != 0)
    }

    pub fn is_true(&self) -> bool {
        self.value != 0x00
    }

    fn get_contents(&self, encoding: &EncodingType) -> Vec<u8> {
        let mut contents = self.value;
        match encoding {
            EncodingType::Der if self.is_true() => {
                contents = 0xFF;
            }
            _ => {}
        }
        vec![contents]
    }
}
impl Display for DerBoolean {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", if self.is_true() { "TRUE" } else { "FALSE" })
    }
}

impl Asn1ObjectInternal for DerBoolean {
    fn get_encoding_with_type(&self, encoding: &EncodingType) -> Box<dyn Asn1Encoding> {
        Box::new(PrimitiveEncoding::new(
            UNIVERSAL,
            BOOLEAN,
            Rc::new(self.get_contents(encoding)),
        ))
    }
}

impl Asn1Convertiable for DerBoolean {
    fn to_asn1_encodable(self: &Rc<Self>) -> Box<dyn Asn1Encodable> {
        Box::new(Asn1ObjectImpl::new(self.clone()))
    }
}
