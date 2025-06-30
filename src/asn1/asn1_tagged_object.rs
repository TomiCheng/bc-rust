use crate::asn1::{asn1_tags, Asn1Object, Asn1OctetString};
use crate::asn1::asn1_tags::{PRIVATE, UNIVERSAL};
use crate::{BcError, Result};


#[derive(Clone, Debug)]
pub struct Asn1TaggedObject {
    explicitness: u8,
    tag_class: u8,
    tag_no: u8,
    object: Asn1Object,
}

impl Asn1TaggedObject {
    const DECLARED_EXPLICIT: u8 = 1;
    const DECLARED_IMPLICIT: u8 = 2;
    const PARSED_EXPLICIT: u8 = 3;
    const PARSED_IMPLICIT: u8 = 4;
    pub(crate) fn new(explicitness: u8, tag_class: u8, tag_no: u8, object: Asn1Object) -> Result<Self> {
        if tag_class == UNIVERSAL || (tag_class & PRIVATE) != tag_class {
            return Err(BcError::with_invalid_argument(format!("invalid tag class: {}", tag_class)));  
        }
        Ok(Self {
            explicitness,
            tag_class,
            tag_no,
            object,
        })
    }
    pub(crate) fn with_context_specific(explicitness: bool, tag_no: u8, object: Asn1Object) -> Result<Self> {
        Self::new(if  explicitness { Self::DECLARED_EXPLICIT } else { Self::DECLARED_IMPLICIT }, asn1_tags::CONTEXT_SPECIFIC, tag_no, object)
    }
    pub(crate) fn create_primitive(tag_class: u8, tag_no: u8, contents_octets: &[u8]) -> Result<Self> {
        Asn1TaggedObject::new(Self::PARSED_IMPLICIT, tag_class, tag_no, 
                              Asn1Object::from(Asn1OctetString::with_contents(contents_octets)))
    }
    pub fn has_tag(&self, tag_class: u8, tag_no: u8) -> bool {
        self.tag_class == tag_class && self.tag_no == tag_no
    }
    pub fn has_tag_class(&self,tag_class: u8) -> bool {
        self.tag_class == tag_class
    }
    pub fn is_explicit(&self) -> bool {
        self.explicitness == Self::DECLARED_EXPLICIT || self.explicitness == Self::PARSED_EXPLICIT
    }
    pub fn is_parsed(&self) -> bool {
        self.explicitness == Self::PARSED_EXPLICIT || self.explicitness == Self::PARSED_IMPLICIT
    }
    pub fn get_base_object(&self) -> &Asn1Object {
        &self.object
    }
    pub fn get_base_universal(&self) -> Asn1Object {
        match self.explicitness {
            ///PARSED_EXPLICIT =>  {},
            //PARSED_IMPLICIT => {
                // if let Some(sequence) = self.object.as_sequence() {
                //     
                // } else {
                //     
                // }
            //},
            _ => {
                self.object.clone()
            }
        }
    }
}