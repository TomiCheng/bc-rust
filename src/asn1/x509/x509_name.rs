use crate::asn1::EncodingType::Ber;
use crate::asn1::pkcs::pkcs_object_identifiers::*;
use crate::asn1::x509::{X509DefaultEntryConverter, X509NameEntryConverter, X509NameTokenizer};
use crate::asn1::x509::x509_object_identifiers::*;
use crate::asn1::{Asn1Encodable, Asn1EncodableVector, Asn1Object, Asn1ObjectIdentifier, Asn1Sequence, Asn1Set, Asn1TaggedObject};
use crate::util::encoders::hex::to_hex_string;
use crate::{GLOBAL, Result, BcError};
use std::collections::HashMap;
use std::fmt;
use std::sync::LazyLock;
use crate::asn1::x500::style::ietf_utilities::unescape;

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
    converter: Box<dyn X509NameEntryConverter>,
}

impl X509Name {
    pub(crate) fn get_tagged(p0: Asn1TaggedObject, p1: bool) -> Result<Self> {
        todo!()
    }
    fn new(ordering: Vec<Asn1ObjectIdentifier>, values: Vec<String>, added: Vec<bool>, converter: Box<dyn X509NameEntryConverter>) -> Self {
        X509Name { ordering, values, added, converter }
    }
    pub fn with_str(dir_name: &str) -> Result<Self> {
        Self::with_reverse_lookup_str((*GLOBAL).x509_name_default_reverse(), &DEFAULT_LOOKUP, dir_name)
    }
    pub fn with_ordering_attributes(
        ordering: Vec<Asn1ObjectIdentifier>,
        attributes: HashMap<Asn1ObjectIdentifier, String>,
    ) -> Result<Self> {
        Self::with_ordering_attributes_converter(ordering, attributes, Box::new(X509DefaultEntryConverter))
    }
    pub fn with_ordering_attributes_converter(
        ordering: Vec<Asn1ObjectIdentifier>,
        attributes: HashMap<Asn1ObjectIdentifier, String>,
        converter: Box<dyn X509NameEntryConverter>,
    ) -> Result<Self> {
        if ordering.len() != attributes.len() {
            return Err(BcError::with_invalid_argument("'oids' must be same length as 'values'."));
        }

        let added = vec![false; ordering.len()];
        Ok(X509Name::new(ordering, attributes.into_values().collect(), added, converter))
    }
    pub fn with_reverse_lookup_str(reverse: bool, lookup: &HashMap<String, Asn1ObjectIdentifier>, dir_name: &str) -> Result<Self> {
        Self::with_reverse_lookup_str_converter(reverse, lookup, dir_name, Box::new(X509DefaultEntryConverter))
    }
    pub fn with_reverse_lookup_str_converter(
        reverse: bool,
        lookup: &HashMap<String, Asn1ObjectIdentifier>,
        dir_name: &str,
        converter: Box<dyn X509NameEntryConverter>,
    ) -> Result<Self> {
        let mut name_tokenizer = X509NameTokenizer::with_str(dir_name);

        let mut result = X509Name::new(Vec::new(), Vec::new(), Vec::new(), converter);

        while let Some(rdn) = name_tokenizer.next_token()? {
            let mut rdn_tokenizer = X509NameTokenizer::with_str_and_separator(rdn, '+')?;
            let token = rdn_tokenizer.next_token()?.ok_or(BcError::with_invalid_argument("badly formatted directory string"))?;
            result.add_attribute(lookup, token, false)?;
            while let Some(token) = rdn_tokenizer.next_token()? {
                result.add_attribute(lookup, token, true)?;
            }
        }

        if reverse {

        }


        Ok(result)
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
        Ok(X509Name::new(ordering, values, added, Box::new(X509DefaultEntryConverter)))
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
            if ava.len() > 0 && !self.added[i] {
                components.push(ava);
                ava = String::new();
            }
            if self.added[i] {
                ava.push('+');
            }
            append_value(&mut ava, oid_symbols, &self.ordering[i], &self.values[i]);
        }
        if !ava.is_empty() {
            components.push(ava);
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
    fn add_attribute(&mut self, lookup: &HashMap<String, Asn1ObjectIdentifier>, token: &str, added: bool) -> Result<()> {
        let mut tokenizer = X509NameTokenizer::with_str_and_separator(token, '=')?;

        let type_token = tokenizer.next_token()?.ok_or(BcError::with_invalid_argument("badly formatted directory string"))?;
        let value_token = tokenizer.next_token()?.ok_or(BcError::with_invalid_argument("badly formatted directory string"))?;

        let oid = Self::decode_oid(type_token.trim(), lookup)?;
        let value = unescape(value_token);

        self.ordering.push(oid);
        self.values.push(value);
        self.added.push(added);

        Ok(())
    }
    fn decode_oid(name: &str, lookup: &HashMap<String, Asn1ObjectIdentifier>) -> Result<Asn1ObjectIdentifier> {
        if name.starts_with("OID.") || name.starts_with("oid.") {
            return Ok(Asn1ObjectIdentifier::with_str(&name[4..])?);
        }

        if let Some(r) = Asn1ObjectIdentifier::try_parse(name) {
            return Ok(r);
        }

        if let Some(r) = lookup.get(name) {
            return Ok(r.clone());
        }

        Err(BcError::with_invalid_argument(format!("unknown object id - {} - passed to distinguished name", name)))
    }
}

impl fmt::Display for X509Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_with_symbols(false, &DEFAULT_SYMBOLS))
    }
}
impl From<X509Name> for Asn1Object {
    fn from(value: X509Name) -> Self {



        //let mut vec = Asn1EncodableVector::new();
        //let mut s_vec = Asn1EncodableVector::new();



        //let sequence = Asn1Sequence::with_
        todo!()
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

static DEFAULT_SYMBOLS: LazyLock<Symbols> = LazyLock::new(|| {
    let mut symbols = Symbols::new();
    symbols.insert(COUNTRY_NAME.clone(), "C".to_owned());
    symbols.insert(ORGANIZATION_NAME.clone(), "O".to_owned());
    symbols.insert(TITLE.clone(), "T".to_owned());
    symbols.insert(ORGANIZATIONAL_UNIT_NAME.clone(), "OU".to_owned());
    symbols.insert(COMMON_NAME.clone(), "CN".to_owned());
    symbols.insert(LOCALITY_NAME.clone(), "L".to_owned());
    symbols.insert(STATE_OR_PROVINCE_NAME.clone(), "ST".to_owned());
    symbols.insert(SERIAL_NUMBER.clone(), "SERIALNUMBER".to_owned());
    symbols.insert(PKCS9_AT_EMAIL_ADDRESS.clone(), "E".to_owned());
    symbols.insert(DC.clone(), "DC".to_owned());
    symbols.insert(UID.clone(), "UID".to_owned());
    symbols.insert(STREET.clone(), "STREET".to_owned());
    symbols.insert(SURNAME.clone(), "SURNAME".to_owned());
    symbols.insert(GIVEN_NAME.clone(), "GIVENNAME".to_owned());
    symbols.insert(INITIALS.clone(), "INITIALS".to_owned());
    symbols.insert(GENERATION.clone(), "GENERATION".to_owned());
    symbols.insert(DESCRIPTION.clone(), "DESCRIPTION".to_owned());
    symbols.insert(ROLE.clone(), "ROLE".to_owned());
    symbols.insert(PKCS9_AT_UNSTRUCTURED_ADDRESS.clone(), "unstructuredAddress".to_owned());
    symbols.insert(PKCS9_AT_UNSTRUCTURED_NAME.clone(), "unstructuredName".to_owned());
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

static DEFAULT_LOOKUP: LazyLock<HashMap<String, Asn1ObjectIdentifier>> = LazyLock::new(|| {
    let mut lookup = HashMap::new();
    lookup.insert("c".to_owned(), COUNTRY_NAME.clone());
    lookup
});

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::LazyLock;
    use crate::asn1::{Asn1Object, Asn1ObjectIdentifier, Asn1Sequence, Asn1Utf8String};
    use crate::asn1::x500::style::ietf_utilities::asn1_object_to_string;
    use crate::asn1::x509::x509_object_identifiers::{COMMON_NAME, DN_QUALIFIER, SERIAL_NUMBER};
    use crate::asn1::x509::X509Name;
    use crate::Result;

    #[test]
    fn test_ietf_utilities() {
        let o = Asn1Utf8String::with_str(" ").into();
        asn1_object_to_string(&o).unwrap();
    }
    #[test]
    fn test_bogus_equals() {
        assert!(X509Name::with_str("CN=foo=bar").is_err());
    }
    #[test]
    fn test_encoding_printable_string() {
        do_test_encoding_printable_string(&(*COMMON_NAME), "AU");
        do_test_encoding_printable_string(&(*SERIAL_NUMBER), "123456");
        do_test_encoding_printable_string(&(*DN_QUALIFIER), "123456");
    }

    fn do_test_encoding_printable_string(oid: &Asn1ObjectIdentifier, value: &str) {
        let converted = create_entry_value(oid, value).unwrap();
        assert!(converted.is_printable_string());
    }

    fn create_entry_value(oid: &Asn1ObjectIdentifier, value: &str) -> Result<Asn1Object> {
        let mut attrs = HashMap::new();
        attrs.insert(oid.clone(), value.to_owned());

        let mut ord = Vec::new();
        ord.push(oid.clone());

        let name = X509Name::with_ordering_attributes(ord, attrs)?;

        let sequence: Asn1Object = name.into();





        Ok(sequence)
    }
}
