use std::collections::HashMap;
use crate::asn1::{Asn1Boolean, Asn1ObjectIdentifier, Asn1OctetString, Asn1Sequence, Asn1TaggedObject};
use crate::asn1::x509::X509Extension;
use crate::{define_oid, Result};
use crate::asn1::try_from_tagged::{TryFromTagged, TryIntoTagged};

pub struct X509Extensions {
    ordering: Vec<Asn1ObjectIdentifier>,
    extensions: HashMap<Asn1ObjectIdentifier, X509Extension>,
}

impl X509Extensions {
    fn new(ordering: Vec<Asn1ObjectIdentifier>, extensions: HashMap<Asn1ObjectIdentifier, X509Extension>) -> Self {
        X509Extensions { ordering, extensions }
    }
    fn from_sequence(sequence: Asn1Sequence) -> Result<Self> {
        let mut ordering = Vec::new();
        let mut extensions = HashMap::new();

        for asn1_object in sequence {
            let s: Asn1Sequence = asn1_object.try_into()?;
            let length = s.len();
            if length < 2 || length > 3 {
                return Err(crate::BcError::with_invalid_argument(format!("Bad sequence size: {}", s.len())));
            }

            let mut iter = s.into_iter();
            let oid = Asn1ObjectIdentifier::from_asn1_object(iter.next().unwrap())?;
            let mut is_critical = false;
            if length == 3 {
                let boolean: Asn1Boolean = iter.next().unwrap().try_into()?;
                is_critical = boolean.is_true();
            }
            let octets = Asn1OctetString::from_asn1_object(iter.next().unwrap())?;

            ordering.push(oid.clone());
            extensions.insert(oid.clone(), X509Extension::new(is_critical, octets));
        }
        Ok(X509Extensions::new(ordering, extensions))
    }
    pub fn iter_ordering(&self) -> impl Iterator<Item = &Asn1ObjectIdentifier> {
        self.ordering.iter()
    }
    /// return the extension represented by the object identifier passed in.
    /// 
    /// # Arguments
    /// * `oid` - The object identifier of the extension to retrieve.
    /// 
    /// # Returns
    /// * `Option<&X509Extension>` - Returns an `Option` containing a reference to the `X509Extension` if it exists, or `None` if it does not.
    pub fn get_extension(&self, oid: &Asn1ObjectIdentifier) -> Option<&X509Extension> {
        self.extensions.get(oid)
    }
}
impl TryFromTagged for X509Extensions {
    fn try_from_tagged(tagged_object: Asn1TaggedObject, declared_explicit: bool) -> Result<Self> {
        let sequence = tagged_object.try_into_tagged(declared_explicit)?;
        Self::from_sequence(sequence)
    }
}


define_oid!(SUBJECT_KEY_IDENTIFIER, "2.5.29.14", "Subject Key Identifier");
define_oid!(KEY_USAGE, "2.5.29.15", "Key Usage");
define_oid!(EXTENDED_KEY_USAGE, "2.5.29.37", "Extended Key Usage");
define_oid!(SUBJECT_ALTERNATIVE_NAME, "2.5.29.17", "Subject Alternative Name");