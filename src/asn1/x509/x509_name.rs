use crate::Result;
use crate::asn1::EncodingType::Ber;
use crate::asn1::x509::x509_object_identifiers;
use crate::asn1::{Asn1Encodable, Asn1Object, Asn1ObjectIdentifier, Asn1Sequence, Asn1Set, Asn1TaggedObject};
use crate::util::encoders::hex::to_hex_string;
use std::collections::HashMap;
use std::fmt;
use std::sync::LazyLock;
use crate::define_oid;
use crate::asn1::pkcs::pkcs_object_identifiers::*;
type Symbols = HashMap<Asn1ObjectIdentifier, String>;

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
    pub(crate) fn get_tagged(p0: Asn1TaggedObject, p1: bool) -> Result<Self> {
        todo!()
    }
}

impl X509Name {
    fn new(ordering: Vec<Asn1ObjectIdentifier>, values: Vec<String>, added: Vec<bool>) -> Self {
        X509Name { ordering, values, added }
    }
    fn from_sequence(sequence: Asn1Sequence) -> Result<Self> {
        let mut ordering = Vec::new();
        let mut values = Vec::new();
        let mut added = Vec::new();
        // RDNSequence ::= SEQUENCE OF RelativeDistinguishedName
        for asn1_object in sequence {
            let rdn_set = Asn1Set::from_asn1_object(asn1_object)?;
            let mut first = true;
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
                added.push(!first);
                if first {
                    first = false;
                }
            }
        }
        Ok(X509Name::new(ordering, values, added))
    }
    /// Convert the structure to a string - if reverse is `true` the
    /// `oids` and values are listed out starting with the last element
    /// in the sequence (ala RFC 2253), otherwise the string will begin
    /// with the first element of the structure. If no string definition
    /// for the oid is found in `oid_symbols` the string value of the oid is
    /// added. Two standard symbol tables are provided DefaultSymbols, and
    /// RFC2253Symbols as part of this class.
    ///
    /// # Arguments
    /// * `reverse` - reverse if true start at the end of the sequence and work back.
    /// * `oid_symbols` - look up table strings for `oids`.
    pub fn to_string_with_symbols(&self, reverse: bool, oid_symbols: &Symbols) -> String {
        let mut components = Vec::new();
        let mut ava = String::new();
        for i in 0..self.ordering.len() {
            if self.added[i] {
                ava.push('+');
                append_value(&mut ava, oid_symbols, &self.ordering[i], &self.values[i]);
            } else {
                ava = String::new();
                append_value(&mut ava, oid_symbols, &self.ordering[i], &self.values[i]);
                components.push(ava.clone());
            }
        }

        if reverse {
            components.reverse();
        }

        let mut result = String::new();
        if !components.is_empty() {
            result.push_str(&components[0]);

            for component in components.iter().skip(1) {
                result.push_str(",");
                result.push_str(component);
            }
        }
        result
    }
}

impl fmt::Display for X509Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_with_symbols(false, &DEFAULT_SYMBOLS))
    }
}
impl From<X509Name> for Asn1Object {
    fn from(value: X509Name) -> Self {
        todo!();
    }
}
impl TryFrom<Asn1Object> for X509Name {
    type Error = crate::BcError;

    fn try_from(asn1_object: Asn1Object) -> Result<Self> {
        if let Ok(sequence) = asn1_object.try_into() {
            return X509Name::from_sequence(sequence);
        }
        todo!();
    }
}
// TODO Refactor common code between this and IetfUtilities.ValueToString
fn append_value(buffer: &mut String, oid_symbols: &Symbols, oid: &Asn1ObjectIdentifier, value: &str) {
    buffer.push_str(oid_symbols.get(oid).unwrap_or(oid.id()));
    buffer.push('=');

    let mut chars = value.chars().peekable();
    let mut first = true;
    let mut end_space = 0;
    while let Some(c) = chars.next() {
        let next_c = chars.peek();

        if c == '\\' && next_c == Some(&'#') {
            buffer.push_str("\\#");
            chars.next(); // consume the '#'
            continue;
        }

        if c == ',' || c == '"' || c == '\\' || c == '+' || c == '=' || c == '<' || c == '>' || c == ';' {
            buffer.push('\\');
        }

        if first && c == ' ' {
           buffer.push('\\');
           buffer.push(c);
           continue;
        } else {
            first = false;
        }

        if c == ' ' {
            end_space += 1;
            continue;
        }

        if end_space > 0 {
            buffer.push_str(&" ".repeat(end_space));
            end_space = 0;
        }

        buffer.push(c);
    }

    if end_space > 0 {
        buffer.push_str(&"\\ ".repeat(end_space));
    }
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
define_oid!(ID_AT_TELEPHONE_NUMBER, ATTRIBUTE_TYPE, "20", "telephone number");
define_oid!(NAME, ATTRIBUTE_TYPE, "41", "Name");
define_oid!(ID_AT_NAME, ATTRIBUTE_TYPE, "41", "Name");
define_oid!(GIVEN_NAME, ATTRIBUTE_TYPE, "42", "given name");
define_oid!(INITIALS, ATTRIBUTE_TYPE, "43", "initials");
define_oid!(GENERATION, ATTRIBUTE_TYPE, "44", "generation");
define_oid!(UNIQUE_IDENTIFIER, ATTRIBUTE_TYPE, "45", "unique identifier");
define_oid!(DN_QUALIFIER, ATTRIBUTE_TYPE, "46", "DN qualifier");
define_oid!(PSEUDONYM, ATTRIBUTE_TYPE, "65", "pseudonym");
define_oid!(ROLE, ATTRIBUTE_TYPE, "72", "role");
define_oid!(ID_AT_ORGANIZATION_IDENTIFIER, ATTRIBUTE_TYPE, "97", "");
define_oid!(DATE_OF_BIRTH, x509_object_identifiers::ID_PDA, "1", "RFC 3039 DateOfBirth - GeneralizedTime - YYYYMMDD000000Z");
define_oid!(PLACE_OF_BIRTH, x509_object_identifiers::ID_PDA, "2", "RFC 3039 PlaceOfBirth - DirectoryString(SIZE(1..128))");
define_oid!(GENDER, x509_object_identifiers::ID_PDA, "3", "RFC 3039 DateOfBirth - PrintableString (SIZE(1)) -- \"M\", \"F\", \"m\" or \"f\"");
define_oid!(COUNTRY_OF_CITIZENSHIP, x509_object_identifiers::ID_PDA, "4", "RFC 3039 CountryOfCitizenship - PrintableString (SIZE (2)) -- ISO 3166");
define_oid!(COUNTRY_OF_RESIDENCE, x509_object_identifiers::ID_PDA, "5", "RFC 3039 CountryOfResidence - PrintableString (SIZE (2)) -- ISO 3166");
define_oid!(NAME_AT_BIRTH, "1.3.36.8.3.14", "ISIS-MTT NameAtBirth - DirectoryString(SIZE(1..64)");

pub static EMAIL_ADDRESS: LazyLock<Asn1ObjectIdentifier> = LazyLock::new(|| PKCS9_AT_EMAIL_ADDRESS.clone());
pub static UNSTRUCTURED_NAME: LazyLock<Asn1ObjectIdentifier> = LazyLock::new(|| PKCS9_AT_UNSTRUCTURED_NAME.clone());
pub static UNSTRUCTURED_ADDRESS: LazyLock<Asn1ObjectIdentifier> = LazyLock::new(|| PKCS9_AT_UNSTRUCTURED_ADDRESS.clone());
pub static ORGANIZATION_IDENTIFIER: LazyLock<Asn1ObjectIdentifier> = LazyLock::new(|| ID_AT_ORGANIZATION_IDENTIFIER.clone());
define_oid!(DC, "0.9.2342.19200300.100.1.25", "others");
define_oid!(UID, "0.9.2342.19200300.100.1.25", "LDAP User id.");
define_oid!(JURISDICTION_L, "1.3.6.1.4.1.311.60.2.1.1", "CA/Browser Forum https://cabforum.org/uploads/CA-Browser-Forum-BR-v2.0.0.pdf, Table 78");
define_oid!(JURISDICTION_ST, "1.3.6.1.4.1.311.60.2.1.2", "CA/Browser Forum https://cabforum.org/uploads/CA-Browser-Forum-BR-v2.0.0.pdf, Table 78");
define_oid!(JURISDICTION_C, "1.3.6.1.4.1.311.60.2.1.3", "CA/Browser Forum https://cabforum.org/uploads/CA-Browser-Forum-BR-v2.0.0.pdf, Table 78");

static DEFAULT_SYMBOLS: LazyLock<Symbols> = LazyLock::new(|| {
    let mut symbols = Symbols::new();
    symbols.insert(C.clone(), "C".to_owned());

    symbols.insert(O.clone(), "O".to_owned());
    symbols.insert(T.clone(), "T".to_owned());
    symbols.insert(OU.clone(), "OU".to_owned());
    symbols.insert(CN.clone(), "CN".to_owned());
    symbols.insert(L.clone(), "L".to_owned());
    symbols.insert(ST.clone(), "ST".to_owned());
    symbols.insert(SERIAL_NUMBER.clone(), "SERIALNUMBER".to_owned());
    symbols.insert(EMAIL_ADDRESS.clone(), "E".to_owned());
    symbols.insert(DC.clone(), "DC".to_owned());
    symbols.insert(UID.clone(), "UID".to_owned());
    symbols.insert(STREET.clone(), "STREET".to_owned());
    symbols.insert(SURNAME.clone(), "SURNAME".to_owned());
    symbols.insert(GIVEN_NAME.clone(), "GIVENNAME".to_owned());
    symbols.insert(INITIALS.clone(), "INITIALS".to_owned());
    symbols.insert(GENERATION.clone(), "GENERATION".to_owned());
    symbols.insert(DESCRIPTION.clone(), "DESCRIPTION".to_owned());
    symbols.insert(ROLE.clone(), "ROLE".to_owned());
    symbols.insert(UNSTRUCTURED_ADDRESS.clone(), "unstructuredAddress".to_owned());
    symbols.insert(UNSTRUCTURED_NAME.clone(), "unstructuredName".to_owned());
    symbols.insert(UNIQUE_IDENTIFIER.clone(), "UniqueIdentifier".to_owned());
    symbols.insert(DN_QUALIFIER.clone(), "DN".to_owned());
    symbols.insert(PSEUDONYM.clone(), "Pseudonym".to_owned());
    symbols.insert(POSTAL_ADDRESS.clone(), "PostalAddress".to_owned());
    symbols.insert(NAME_AT_BIRTH.clone(), "NameAtBirth".to_owned());
    symbols.insert(COUNTRY_OF_CITIZENSHIP.clone(), "CountryOfCitizenship".to_owned());
    symbols.insert(COUNTRY_OF_RESIDENCE.clone(), "CountryOfResidence".to_owned());
    symbols.insert(GENDER.clone(), "Gender".to_owned());
    symbols.insert(PLACE_OF_BIRTH.clone(), "PlaceOfBirth".to_owned());
    symbols.insert(DATE_OF_BIRTH.clone(), "DateOfBirth".to_owned());
    symbols.insert(POSTAL_CODE.clone(), "PostalCode".to_owned());
    symbols.insert(BUSINESS_CATEGORY.clone(), "BusinessCategory".to_owned());
    symbols.insert(TELEPHONE_NUMBER.clone(), "TelephoneNumber".to_owned());
    symbols.insert(NAME.clone(), "Name".to_owned());
    symbols.insert(ORGANIZATION_IDENTIFIER.clone(), "organizationIdentifier".to_owned());
    symbols.insert(JURISDICTION_C.clone(), "jurisdictionCountry".to_owned());
    symbols.insert(JURISDICTION_ST.clone(), "jurisdictionState".to_owned());
    symbols.insert(JURISDICTION_L.clone(), "jurisdictionLocality".to_owned());

    symbols
});
