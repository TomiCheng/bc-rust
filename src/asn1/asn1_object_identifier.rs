use std::process::id;
use crate::asn1::Asn1Object;
use crate::{BcError, Result};
use crate::asn1::asn1_relative_oid::{is_valid_identifier, Asn1RelativeOid};

#[derive(Clone, Debug)]
pub struct Asn1ObjectIdentifier {
    contents: Vec<u8>,
    
}

impl Asn1ObjectIdentifier {
    const MAX_IDENTIFIER_LENGTH: usize = 4096;
    pub fn new(contents: Vec<u8>) -> Self {
        Asn1ObjectIdentifier { contents }
    }
    pub fn with_str(identifier: &str) -> Result<Self> {
        Self::check_identifier(identifier)?;
        todo!();
    }
    pub(crate) fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        // todo!()
        Ok(Asn1ObjectIdentifier { contents })
    }
    pub(crate) fn from_asn1_object(asn1_object: &Asn1Object) -> Result<Self> {
        if let Some(object_identifier) = asn1_object.as_object_identifier() {
            Ok(object_identifier.clone())
        } else {
            Err(crate::BcError::with_invalid_cast("Expected an octet string for Asn1ObjectIdentifier"))
        }
    }
    fn check_identifier(identifier: &str) -> Result<()> {
        if identifier.is_empty() {
            return Err(crate::BcError::with_invalid_argument("Identifier cannot be empty"));
        } else if identifier.len() > Self::MAX_IDENTIFIER_LENGTH {
            return Err(crate::BcError::with_invalid_argument(format!("Identifier exceeds maximum length of {} characters", Self::MAX_IDENTIFIER_LENGTH)));
        } else if !Self::is_valid_identifier(identifier) {
            return Err(BcError::with_invalid_format("Identifier contains invalid characters"));
        }
        Ok(())
    }
    fn is_valid_identifier(identifier: &str) -> bool {
        let count = identifier.chars().count();
        let mut chars = identifier.chars();
        
        if count < 3 || chars.nth(1) != Some('.') {
            return false;
        }
        
        if chars.nth(1) != Some('.') {
            return false;
        }
        
        let first = chars.nth(0);
        if first < Some('0') || first > Some('2') {
            return false;
        }
        // let d = identifier.chars().skip(2);
        // if !is_valid_identifier(d) {
        //     return false;
        // }
        
        if first == Some('2') {
            return true;
        }
        
        if count == 3 || chars.nth(3) == Some('.') {
            return true;
        }
        
        if count == 4 && chars.nth(4) == Some('.') {
            return chars.nth(2) < Some('4');
        }
        
        false
    }
}