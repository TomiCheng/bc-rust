use std::sync;

use crate::asn1;
use crate::Result;
use super::x509_object_identifiers;


macro_rules! define_oid {
    ($name:ident, $oid:expr) => {
        pub static $name: sync::LazyLock<asn1::DerObjectIdentifier> =
        sync::LazyLock::new(|| asn1::DerObjectIdentifier::with_str($oid).unwrap());
    };
    ($name:ident, $oid:expr, $doc:literal) => {
        #[doc = $doc]
        pub static $name: sync::LazyLock<asn1::DerObjectIdentifier> =
        sync::LazyLock::new(|| asn1::DerObjectIdentifier::with_str($oid).unwrap());
    };
    ($name:ident, $base:expr, $branch:expr) => {
        pub static $name: sync::LazyLock<asn1::DerObjectIdentifier> =
        sync::LazyLock::new(|| $base.branch($branch).unwrap());
    };
    ($name:ident, $base:expr, $branch:expr, $doc:literal) => {
        #[doc = $doc]
        pub static $name: sync::LazyLock<asn1::DerObjectIdentifier> =
        sync::LazyLock::new(|| $base.branch($branch).unwrap());
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

define_oid!(DATE_OF_BIRTH, x509_object_identifiers::ID_PDA, "1", "RFC 3039 DateOfBirth - GeneralizedTime - YYYYMMDD000000Z");
define_oid!(PLACE_OF_BIRTH, x509_object_identifiers::ID_PDA, "2", "RFC 3039 PlaceOfBirth - DirectoryString(SIZE(1..128))");
define_oid!(GENDER, x509_object_identifiers::ID_PDA, "3", "RFC 3039 DateOfBirth - PrintableString (SIZE(1)) -- \"M\", \"F\", \"m\" or \"f\"");
define_oid!(COUNTRY_OF_CITIZENSHIP, x509_object_identifiers::ID_PDA, "4", "RFC 3039 CountryOfCitizenship - PrintableString (SIZE (2)) -- ISO 3166");
define_oid!(COUNTRY_OF_RESIDENCE, x509_object_identifiers::ID_PDA, "5", "RFC 3039 CountryOfResidence - PrintableString (SIZE (2)) -- ISO 3166");


pub struct X509Name {
    values: sync::Arc<Vec<String>>,
}

impl X509Name {
    pub fn with_asn1_sequence(sequence: asn1::Asn1Sequence) -> Result<Self> {
        for asn1_object in sequence.iter() {
            //asn1_object.to_asn1_object().as_any().downcast_ref::<asn1::Asn1Set>();
        }
        todo!()
    }
}
// trait

impl Default for X509Name {
    fn default() -> Self {
        X509Name {
            values: sync::Arc::new(Vec::new()),
        }
    }
}