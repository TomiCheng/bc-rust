use crate::Result;
use crate::asn1::{Asn1Encodable, Asn1Object, Asn1ObjectIdentifier, Asn1Sequence, Asn1Set};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};

/// ```text
/// RDNSequence ::= SEQUENCE OF RelativeDistinguishedName
///
/// RelativeDistinguishedName ::= SET SIZE (1..MAX) OF AttributeTypeAndValue
///
/// AttributeTypeAndValue ::= SEQUENCE {
///    type AttributeType,
///    value AttributeValue
/// }
/// ```
pub struct X509Name {
    ordering: Vec<Asn1ObjectIdentifier>,
    values: Vec<String>,
    added: Vec<bool>,
}

impl X509Name {
    fn new(ordering: Vec<Asn1ObjectIdentifier>, values: Vec<String>, added: Vec<bool>) -> Self {
        X509Name { ordering, values, added }
    }
    fn from_sequence(sequence: Asn1Sequence) -> Result<Self> {
        let mut ordering = Vec::new();
        let mut values = Vec::new();
        // RDNSequence ::= SEQUENCE OF RelativeDistinguishedName
        for asn1_object in sequence {
            let rdn_set = Asn1Set::from_asn1_object(asn1_object)?;
            for attribute_type_and_value in rdn_set {
                let attribute_type_and_value: Asn1Sequence = attribute_type_and_value.try_into()?;
                if attribute_type_and_value.len() != 2 {
                    return Err(crate::BcError::with_invalid_format("badly sized AttributeTypeAndValue"));
                }
                let mut iter = attribute_type_and_value.into_iter();
                let type_object = iter.next().unwrap();
                let value_object = iter.next().unwrap();

                ordering.push(type_object.try_into()?);
                if let Some(string_object) = value_object.as_string()
                    && !value_object.is_universal_string()
                {
                    let mut v = string_object.to_asn1_string()?;
                    if v.starts_with("#") {
                        v = format!("\\{}", v);
                    }
                    values.push(v);
                } else {
                    values.push(format!("#{}", to_hex_string(&(value_object.get_encoded(Ber)?))));
                }
            }
        }
        Ok(X509Name::new(ordering, Vec::new(), Vec::new()))
    }
    pub(crate) fn with_asn1_object(asn1_object: Asn1Object) -> Result<Self> {
        if let Ok(sequence) = asn1_object.try_into() {
            return X509Name::from_sequence(sequence);
        }
        //get_instance_choice()
        todo!()
    }
    pub fn to_string_(&self, reverse: bool, oid_symbols: HashMap<Asn1ObjectIdentifier, String>) -> String {
        //let mut result = Vec::new();
        for order in &self.ordering {
            
        }
        
        todo!();
    }
}

impl Display for X509Name {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        //self.to_string_()
        todo!();
    }
}

use crate::asn1;
use crate::asn1::EncodingType::Ber;
use crate::asn1::x509::x509_object_identifiers;
use crate::util::encoders::hex::to_hex_string;
use std::sync;
use std::sync::{Arc, LazyLock};

macro_rules! define_oid {
    ($name:ident, $oid:expr) => {
        pub static $name: LazyLock<Arc<Asn1ObjectIdentifier>> = LazyLock::new(|| Arc::new(Asn1ObjectIdentifier::with_str($oid).unwrap()));
    };
    ($name:ident, $oid:expr, $doc:literal) => {
        #[doc = $doc]
        pub static $name: LazyLock<Arc<Asn1ObjectIdentifier>> = LazyLock::new(|| Arc::new(Asn1ObjectIdentifier::with_str($oid).unwrap()));
    };
    ($name:ident, $base:expr, $branch:expr) => {
        pub static $name: LazyLock<Arc<Asn1ObjectIdentifier>> = LazyLock::new(|| Arc::new($base.branch($branch).unwrap()));
    };
    ($name:ident, $base:expr, $branch:expr, $doc:literal) => {
        #[doc = $doc]
        pub static $name: LazyLock<Arc<Asn1ObjectIdentifier>> = LazyLock::new(|| Arc::new($base.branch($branch).unwrap()));
    };
}
// 定義基礎 OID
define_oid!(ATTRIBUTE_TYPE, "2.5.4", "X.500 attribute type base OID");

define_oid!(CN, ATTRIBUTE_TYPE, "3", "common name - StringType(SIZE(1..64))");
define_oid!(SURNAME, ATTRIBUTE_TYPE, "4", "surname");
define_oid!(SERIAL_NUMBER, ATTRIBUTE_TYPE, "5", "device serial number name - StringType(SIZE(1..64))");
define_oid!(C, ATTRIBUTE_TYPE, "6", "country name - StringType(SIZE(2))");
define_oid!(L, ATTRIBUTE_TYPE, "7", "locality name - StringType(SIZE(1..64))");
define_oid!(ST, ATTRIBUTE_TYPE, "8", "state, or province name - StringType(SIZE(1..64))");
define_oid!(STREET, ATTRIBUTE_TYPE, "9", "street - StringType(SIZE(1..64))");
define_oid!(O, ATTRIBUTE_TYPE, "10", "organization - StringType(SIZE(1..64))");
define_oid!(OU, ATTRIBUTE_TYPE, "11", "organizational unit name - StringType(SIZE(1..64))");
define_oid!(T, ATTRIBUTE_TYPE, "12", "title");
define_oid!(DESCRIPTION, ATTRIBUTE_TYPE, "13", "description");
define_oid!(SEARCH_GUIDE, ATTRIBUTE_TYPE, "14", "search guide");
define_oid!(BUSINESS_CATEGORY, ATTRIBUTE_TYPE, "15", "businessCategory - DirectoryString(SIZE(1..128)");
define_oid!(POSTAL_ADDRESS, ATTRIBUTE_TYPE, "16", "postal address");
define_oid!(POSTAL_CODE, ATTRIBUTE_TYPE, "17", "postal code - DirectoryString(SIZE(1..40)");
define_oid!(TELEPHONE_NUMBER, ATTRIBUTE_TYPE, "20", "telephone number");
define_oid!(NAME, ATTRIBUTE_TYPE, "41", "Name");
define_oid!(GIVEN_NAME, ATTRIBUTE_TYPE, "42", "given name");
define_oid!(INITIALS, ATTRIBUTE_TYPE, "43", "initials");
define_oid!(GENERATION, ATTRIBUTE_TYPE, "44", "generation");
define_oid!(UNIQUE_IDENTIFIER, ATTRIBUTE_TYPE, "45", "unique identifier");
define_oid!(DN_QUALIFIER, ATTRIBUTE_TYPE, "46", "DN qualifier");
define_oid!(PSEUDONYM, ATTRIBUTE_TYPE, "65", "pseudonym");
define_oid!(ROLE, ATTRIBUTE_TYPE, "72", "role");
define_oid!(
    DATE_OF_BIRTH,
    x509_object_identifiers::ID_PDA,
    "1",
    "RFC 3039 DateOfBirth - GeneralizedTime - YYYYMMDD000000Z"
);
define_oid!(
    PLACE_OF_BIRTH,
    x509_object_identifiers::ID_PDA,
    "2",
    "RFC 3039 PlaceOfBirth - DirectoryString(SIZE(1..128))"
);
define_oid!(
    GENDER,
    x509_object_identifiers::ID_PDA,
    "3",
    "RFC 3039 DateOfBirth - PrintableString (SIZE(1)) -- \"M\", \"F\", \"m\" or \"f\""
);
define_oid!(
    COUNTRY_OF_CITIZENSHIP,
    x509_object_identifiers::ID_PDA,
    "4",
    "RFC 3039 CountryOfCitizenship - PrintableString (SIZE (2)) -- ISO 3166"
);
define_oid!(
    COUNTRY_OF_RESIDENCE,
    x509_object_identifiers::ID_PDA,
    "5",
    "RFC 3039 CountryOfResidence - PrintableString (SIZE (2)) -- ISO 3166"
);

static DEFAULT_SYMBOLS: LazyLock<HashMap<Arc<Asn1ObjectIdentifier>, &'static str>> = LazyLock::new(|| {
    let mut symbols = HashMap::new();
    symbols.insert(C.clone(), "C");
    symbols.insert(O.clone(), "O");

    symbols
});
