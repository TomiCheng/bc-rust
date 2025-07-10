use crate::asn1::EncodingType::Ber;
use crate::asn1::asn1_utilities::try_from_choice_tagged;
use crate::asn1::pkcs::pkcs_object_identifiers::*;
use crate::asn1::try_from_tagged::TryFromTagged;
use crate::asn1::x500::style::ietf_utilities::{escape_dn_string, unescape};
use crate::asn1::x509::x509_object_identifiers::*;
use crate::asn1::x509::{X509DefaultEntryConverter, X509NameEntryConverter, X509NameTokenizer};
use crate::asn1::{Asn1Encodable, Asn1Object, Asn1ObjectIdentifier, Asn1Sequence, Asn1Set, Asn1TaggedObject};
use crate::util::encoders::hex::{to_decode_with_str, to_hex_string};
use crate::{BcError, GLOBAL, Result};
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::LazyLock;

type Symbols = HashMap<Asn1ObjectIdentifier, String>;
type LookupType = HashMap<String, Asn1ObjectIdentifier>;
type ConverterType = Rc<dyn X509NameEntryConverter>;
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
#[derive(Clone)]
pub struct X509Name {
    ordering: Vec<Asn1ObjectIdentifier>,
    values: Vec<String>,
    added: Vec<bool>,
    converter: Rc<dyn X509NameEntryConverter>,
}

impl X509Name {
    fn new(ordering: Vec<Asn1ObjectIdentifier>, values: Vec<String>, added: Vec<bool>, converter: ConverterType) -> Self {
        X509Name {
            ordering,
            values,
            added,
            converter,
        }
    }
    pub fn with_str(dir_name: &str) -> Result<Self> {
        Self::with_reverse_lookup_str((*GLOBAL).x509_name_default_reverse(), &DEFAULT_LOOKUP, dir_name)
    }
    pub fn with_ordering_attributes(ordering: Vec<Asn1ObjectIdentifier>, attributes: Symbols) -> Result<Self> {
        Self::with_ordering_attributes_converter(ordering, attributes, Rc::new(X509DefaultEntryConverter))
    }
    pub fn with_ordering_attributes_converter(ordering: Vec<Asn1ObjectIdentifier>, attributes: Symbols, converter: ConverterType) -> Result<Self> {
        let mut r_ordering = Vec::with_capacity(ordering.len());
        let mut r_values = Vec::with_capacity(ordering.len());
        let mut r_added = Vec::with_capacity(ordering.len());

        let iter = ordering.into_iter();

        for o in iter {
            if let Some(v) = attributes.get(&o) {
                r_ordering.push(o.clone());
                r_values.push(v.clone());
                r_added.push(false);
            } else {
                return Err(BcError::with_invalid_argument(format!(
                    "not attribute for object id - {} - passed to distinguished name",
                    o
                )));
            }
        }
        Ok(X509Name::new(r_ordering, r_values, r_added, converter))
    }
    pub fn with_reverse_lookup_str(reverse: bool, lookup: &LookupType, dir_name: &str) -> Result<Self> {
        Self::with_reverse_lookup_str_converter(reverse, lookup, dir_name, Rc::new(X509DefaultEntryConverter))
    }
    pub fn with_reverse_lookup_str_converter(reverse: bool, lookup: &LookupType, dir_name: &str, converter: ConverterType) -> Result<Self> {
        let mut name_tokenizer = X509NameTokenizer::with_str(dir_name);

        let mut result = X509Name::new(Vec::new(), Vec::new(), Vec::new(), converter);

        while let Some(rdn) = name_tokenizer.next_token()? {
            let mut rdn_tokenizer = X509NameTokenizer::with_str_and_separator(rdn, '+')?;
            let token = rdn_tokenizer
                .next_token()?
                .ok_or(BcError::with_invalid_argument("badly formatted directory string"))?;
            result.add_attribute(lookup, token, false)?;
            while let Some(token) = rdn_tokenizer.next_token()? {
                result.add_attribute(lookup, token, true)?;
            }
        }

        if reverse {}

        Ok(result)
    }
    pub fn with_ordering_values(ordering: Vec<Asn1ObjectIdentifier>, values: Vec<String>) -> Result<Self> {
        Self::with_ordering_values_converter(ordering, values, Rc::new(X509DefaultEntryConverter))
    }
    /// Takes two vectors one of the oids and the other of the values.
    /// The passed in converter will be used to convert the strings into their ASN.1 counterparts.
    pub fn with_ordering_values_converter(ordering: Vec<Asn1ObjectIdentifier>, values: Vec<String>, converter: ConverterType) -> Result<Self> {
        if ordering.len() != values.len() {
            return Err(BcError::with_invalid_argument("'oids' must be same length as 'values'."));
        }

        let mut r_ordering = Vec::with_capacity(ordering.len());
        let mut r_values = Vec::with_capacity(ordering.len());
        let mut r_added = Vec::with_capacity(ordering.len());

        for (o, v) in ordering.into_iter().zip(values.into_iter()) {
            r_ordering.push(o);
            r_values.push(v);
            r_added.push(false);
        }
        Ok(X509Name::new(r_ordering, r_values, r_added, converter))
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
                    return Err(BcError::with_invalid_format("badly sized AttributeTypeAndValue"));
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
        Ok(X509Name::new(ordering, values, added, Rc::new(X509DefaultEntryConverter)))
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

        let type_token = tokenizer
            .next_token()?
            .ok_or(BcError::with_invalid_argument("badly formatted directory string"))?;
        let value_token = tokenizer
            .next_token()?
            .ok_or(BcError::with_invalid_argument("badly formatted directory string"))?;

        let oid = Self::decode_oid(type_token.trim(), lookup)?;
        let value = unescape(value_token);

        self.ordering.push(oid);
        self.values.push(value);
        self.added.push(added);

        if tokenizer.has_more_tokens() {
            return Err(BcError::with_invalid_argument("badly formatted directory string - too many tokens"));
        }

        Ok(())
    }
    fn decode_oid(name: &str, lookup: &HashMap<String, Asn1ObjectIdentifier>) -> Result<Asn1ObjectIdentifier> {
        if name.starts_with("OID.") || name.starts_with("oid.") {
            return Ok(Asn1ObjectIdentifier::with_str(&name[4..])?);
        }

        if let Some(r) = Asn1ObjectIdentifier::try_parse(name) {
            return Ok(r);
        }

        if let Some(r) = lookup.get(&name.to_lowercase()) {
            return Ok(r.clone());
        }

        Err(BcError::with_invalid_argument(format!(
            "unknown object id - {} - passed to distinguished name",
            name
        )))
    }
    /// Check if this name is equivalent to the other name.
    ///
    /// # Arguments
    /// * `other` - The X509Name object to test equivalency against.
    /// * `in_order` - If true, the order of elements must be the same, as well as the values associated with each element.
    ///
    /// # Returns
    /// `true` if the names are equivalent, `false` otherwise.
    pub fn equivalent_in_order(&self, other: &Self, in_order: bool) -> bool {
        if !in_order {
            return self.equivalent(other);
        }
        let ordering_len = self.ordering.len();
        if ordering_len != other.ordering.len() {
            return false;
        }

        let mut iter = self.ordering.iter().zip(self.values.iter());
        let mut other_iter = other.ordering.iter().zip(other.values.iter());
        while let Some((o, v)) = iter.next()
            && let Some((other_o, other_v)) = other_iter.next()
        {
            if o != other_o {
                return false;
            }
            if !Self::equivalent_string(v, other_v) {
                return false;
            }
        }
        true
    }
    /// test for equivalence - note: case is ignored.
    pub fn equivalent(&self, other: &Self) -> bool {
        let ordering_len = self.ordering.len();
        if ordering_len != other.ordering.len() {
            return false;
        }

        if ordering_len == 0 {
            return true;
        }

        let mut indexes = vec![false; ordering_len];
        let start;
        let end;
        let delta;

        if self.ordering[0] == other.ordering[0] {
            // guess forward
            start = 0;
            end = ordering_len as isize;
            delta = 1;
        } else {
            // guess reversed - most common problem
            start = ordering_len - 1;
            end = -1;
            delta = -1;
        }

        let mut i = start as isize;
        while i != end {
            let oid = &self.ordering[i as usize];
            let value = &self.values[i as usize];
            let mut found = false;

            for j in 0..ordering_len {
                if indexes[j] {
                    continue;
                }

                if oid == &other.ordering[j] {
                    if Self::equivalent_string(value, &other.values[j]) {
                        indexes[j] = true;
                        found = true;
                        break;
                    }
                }
            }

            if !found {
                return false;
            }

            i += delta;
        }
        true
    }
    fn equivalent_string(s1: &str, s2: &str) -> bool {
        if s1 == s2 {
            return true;
        }

        let c1 = Self::canonicalize(s1);
        let c2 = Self::canonicalize(s2);

        if c1 == c2 {
            return true;
        }

        let v1 = Self::strip_internal_space(&c1);
        let v2 = Self::strip_internal_space(&c2);
        if v1 == v2 {
            return true;
        }

        false
    }
    fn canonicalize(s: &str) -> String {
        let mut v = s.to_lowercase().trim().to_string();
        if v.starts_with('#') {
            if let Ok(obj) = Self::decode_object(&v) {
                if let Some(obj_str) = obj.as_string() {
                    v = obj_str.to_asn1_string().unwrap().to_lowercase().trim().to_string();
                }
            }
        }
        v.to_string()
    }
    fn decode_object(s: &str) -> Result<Asn1Object> {
        let buffer = to_decode_with_str(s)?;
        Asn1Object::with_bytes(&buffer)
    }
    fn strip_internal_space(s: &str) -> String {
        let mut result = String::new();

        if !s.is_empty() {
            let mut iter = s.chars().peekable();
            let mut c1 = iter.next().unwrap();
            result.push(c1);

            while let Some(c2) = iter.next() {
                if !(c1 == ' ' && c2 == ' ') {
                    result.push(c2);
                }
                c1 = c2;
            }
        }
        result
    }
    pub fn ordering(&self) -> &[Asn1ObjectIdentifier] {
        &self.ordering
    }
    pub fn values(&self) -> &[String] {
        &self.values
    }
}
impl fmt::Display for X509Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_with_symbols(false, &DEFAULT_SYMBOLS))
    }
}
impl Hash for X509Name {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let asn1_object: Asn1Object = self.clone().into();
        asn1_object.hash(state)
    }
}
impl From<X509Name> for Asn1Object {
    fn from(value: X509Name) -> Self {
        let mut ordering = value.ordering.into_iter();
        let mut values = value.values.into_iter();
        let mut added = value.added.into_iter();
        let converter = value.converter;

        let mut has_oid = false;
        let mut vec: Vec<Asn1Object> = Vec::new();
        let mut s_vec: Vec<Asn1Object> = Vec::new();

        while let Some(o) = ordering.next()
            && let Some(v) = values.next()
            && let Some(a) = added.next()
        {
            if !a && has_oid {
                vec.push(Asn1Set::new(s_vec).into());
                s_vec = Vec::new();
            }

            has_oid = true;
            let converted_value = converter.get_converted_value(&o, &v).unwrap();
            let sequence = Asn1Sequence::new(vec![o.into(), converted_value]);
            s_vec.push(sequence.into());
        }

        vec.push(Asn1Set::new(s_vec).into());
        let sequence = Asn1Sequence::new(vec);
        sequence.into()
    }
}
impl TryFrom<Asn1Object> for X509Name {
    type Error = BcError;

    fn try_from(asn1_object: Asn1Object) -> Result<Self> {
        if let Ok(sequence) = asn1_object.try_into() {
            return X509Name::from_sequence(sequence);
        }
        Err(BcError::with_invalid_format(
            "X509Name must be a sequence of sets of AttributeTypeAndValue",
        ))
    }
}
impl TryFromTagged for X509Name {
    fn try_from_tagged(tagged: Asn1TaggedObject, declared_explicit: bool) -> Result<Self>
    where
        Self: Sized,
    {
        try_from_choice_tagged(tagged, declared_explicit, X509Name::try_from)
    }
}
fn append_value(buffer: &mut String, oid_symbols: &Symbols, oid: &Asn1ObjectIdentifier, value: &str) {
    buffer.push_str(oid_symbols.get(oid).unwrap_or(oid.id()));
    buffer.push('=');
    escape_dn_string(value, buffer)
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
    lookup.insert("o".to_owned(), ORGANIZATION_NAME.clone());
    lookup.insert("t".to_owned(), TITLE.clone());
    lookup.insert("ou".to_owned(), ORGANIZATIONAL_UNIT_NAME.clone());
    lookup.insert("cn".to_owned(), COUNTRY_NAME.clone());
    lookup.insert("l".to_owned(), LOCALITY_NAME.clone());
    lookup.insert("st".to_owned(), STREET.clone());
    lookup.insert("sn".to_owned(), SURNAME.clone());
    lookup.insert("serialnumber".to_owned(), SERIAL_NUMBER.clone());
    lookup.insert("street".to_owned(), STREET.clone());
    lookup.insert("emailaddress".to_owned(), PKCS9_AT_EMAIL_ADDRESS.clone());
    lookup.insert("dc".to_owned(), DC.clone());
    lookup.insert("e".to_owned(), PKCS9_AT_EMAIL_ADDRESS.clone());
    lookup.insert("uid".to_owned(), UID.clone());
    lookup.insert("surname".to_owned(), SURNAME.clone());
    lookup.insert("givenname".to_owned(), GIVEN_NAME.clone());
    lookup.insert("initials".to_owned(), INITIALS.clone());
    lookup.insert("generation".to_owned(), GENERATION.clone());
    lookup.insert("description".to_owned(), DESCRIPTION.clone());
    lookup.insert("role".to_owned(), ROLE.clone());
    lookup.insert("unstructuredaddress".to_owned(), PKCS9_AT_UNSTRUCTURED_ADDRESS.clone());
    lookup.insert("unstructuredname".to_owned(), PKCS9_AT_UNSTRUCTURED_NAME.clone());
    lookup.insert("uniqueidentifier".to_owned(), UNIQUE_IDENTIFIER.clone());
    lookup.insert("dn".to_owned(), DN_QUALIFIER.clone());
    lookup.insert("pseudonym".to_owned(), PSEUDONYM.clone());
    lookup.insert("postaladdress".to_owned(), POSTAL_ADDRESS.clone());
    lookup.insert("nameatbirth".to_owned(), NAME_AT_BIRTH.clone());
    lookup.insert("countryofcitizenship".to_owned(), COUNTRY_OF_CITIZENSHIP.clone());
    lookup.insert("countryofresidence".to_owned(), COUNTRY_OF_RESIDENCE.clone());
    lookup.insert("gender".to_owned(), GENDER.clone());
    lookup.insert("placeofbirth".to_owned(), PLACE_OF_BIRTH.clone());
    lookup.insert("dateofbirth".to_owned(), DATE_OF_BIRTH.clone());
    lookup.insert("postalcode".to_owned(), POSTAL_CODE.clone());
    lookup.insert("businesscategory".to_owned(), BUSINESS_CATEGORY.clone());
    lookup.insert("telephonenumber".to_owned(), TELEPHONE_NUMBER.clone());
    lookup.insert("name".to_owned(), ID_AT_NAME.clone());
    lookup.insert("organizationidentifier".to_owned(), ORGANIZATION_IDENTIFIER.clone());
    lookup.insert("jurisdictioncountry".to_owned(), JURISDICTION_C.clone());
    lookup.insert("jurisdictionstate".to_owned(), JURISDICTION_ST.clone());
    lookup.insert("jurisdictionlocality".to_owned(), JURISDICTION_L.clone());
    lookup
});

#[cfg(test)]
mod tests {
    use crate::Result;
    use crate::asn1::pkcs::pkcs_object_identifiers::PKCS9_AT_EMAIL_ADDRESS;
    use crate::asn1::x500::style::ietf_utilities::asn1_object_to_string;
    use crate::asn1::x509::X509Name;
    use crate::asn1::x509::x509_object_identifiers::{
        COMMON_NAME, DATE_OF_BIRTH, DC, DN_QUALIFIER, LOCALITY_NAME, ORGANIZATION_NAME, SERIAL_NUMBER, STREET,
    };
    use crate::asn1::{Asn1Object, Asn1ObjectIdentifier, Asn1Sequence, Asn1Set, Asn1Utf8String};
    use std::collections::HashMap;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

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
    #[test]
    fn test_encoding_ia5_string() {
        do_test_encoding_ia5_string(&(*PKCS9_AT_EMAIL_ADDRESS), "test@test.com");
        do_test_encoding_ia5_string(&(*DC), "test");
    }
    #[test]
    fn test_encoding_generalized_time() {
        do_test_encoding_generalized_time(&(*DATE_OF_BIRTH), "#180F32303032303132323132323232305A");
        do_test_encoding_generalized_time(&(*DATE_OF_BIRTH), "20020122122220Z");
    }
    #[test]
    fn test_composite_01() {
        let mut attrs = HashMap::new();
        attrs.insert((*COMMON_NAME).clone(), "AU".to_owned());
        attrs.insert((*ORGANIZATION_NAME).clone(), "The Legion of the Bouncy Castle".to_owned());
        attrs.insert((*LOCALITY_NAME).clone(), "Melbourne".to_owned());
        attrs.insert((*STREET).clone(), "Victoria".to_owned());
        attrs.insert((*PKCS9_AT_EMAIL_ADDRESS).clone(), "feedback-crypto@bouncycastle.org".to_owned());

        let mut order = Vec::new();
        order.push((*COMMON_NAME).clone());
        order.push((*ORGANIZATION_NAME).clone());
        order.push((*LOCALITY_NAME).clone());
        order.push((*STREET).clone());
        order.push((*PKCS9_AT_EMAIL_ADDRESS).clone());

        let name1 = X509Name::with_ordering_attributes(order.clone(), attrs.clone()).unwrap();
        assert!(name1.equivalent(&name1));
        assert!(name1.equivalent_in_order(&name1, true));

        let name2 = X509Name::with_ordering_attributes(order.clone(), attrs.clone()).unwrap();
        assert!(name1.equivalent(&name2));
        assert!(name1.equivalent_in_order(&name2, true));

        let mut hasher = DefaultHasher::new();
        name1.hash(&mut hasher);
        let hash1 = hasher.finish();

        let mut hasher = DefaultHasher::new();
        name2.hash(&mut hasher);
        let hash2 = hasher.finish();
        assert_eq!(hash1, hash2);
    }
    #[test]
    fn test_composite_02() {
        let mut attrs = HashMap::new();
        attrs.insert((*COMMON_NAME).clone(), "AU".to_owned());
        attrs.insert((*ORGANIZATION_NAME).clone(), "The Legion of the Bouncy Castle".to_owned());
        attrs.insert((*LOCALITY_NAME).clone(), "Melbourne".to_owned());
        attrs.insert((*STREET).clone(), "Victoria".to_owned());
        attrs.insert((*PKCS9_AT_EMAIL_ADDRESS).clone(), "feedback-crypto@bouncycastle.org".to_owned());

        let mut order1 = Vec::new();
        order1.push((*COMMON_NAME).clone());
        order1.push((*ORGANIZATION_NAME).clone());
        order1.push((*LOCALITY_NAME).clone());
        order1.push((*STREET).clone());
        order1.push((*PKCS9_AT_EMAIL_ADDRESS).clone());

        let mut order2 = Vec::new();
        order2.push((*PKCS9_AT_EMAIL_ADDRESS).clone());
        order2.push((*STREET).clone());
        order2.push((*LOCALITY_NAME).clone());
        order2.push((*ORGANIZATION_NAME).clone());
        order2.push((*COMMON_NAME).clone());

        let name1 = X509Name::with_ordering_attributes(order1.clone(), attrs.clone()).unwrap();
        let name2 = X509Name::with_ordering_attributes(order2.clone(), attrs.clone()).unwrap();
        assert!(name1.equivalent(&name2));
        assert!(name1.equivalent_in_order(&name2, false));
        assert!(!name1.equivalent_in_order(&name2, true));

        let oids = name1.ordering();
        assert_eq!(&order1, oids);
    }
    #[test]
    fn test_composite_03() {
        let mut attrs = Vec::new();
        attrs.push("AU".to_owned());
        attrs.push("The Legion of the Bouncy Castle".to_owned());
        attrs.push("Melbourne".to_owned());
        attrs.push("Victoria".to_owned());
        attrs.push("feedback-crypto@bouncycastle.org".to_owned());

        let mut order1 = Vec::new();
        order1.push((*COMMON_NAME).clone());
        order1.push((*ORGANIZATION_NAME).clone());
        order1.push((*LOCALITY_NAME).clone());
        order1.push((*STREET).clone());
        order1.push((*PKCS9_AT_EMAIL_ADDRESS).clone());

        let name1 = X509Name::with_ordering_values(order1.clone(), attrs.clone()).unwrap();
        let values = name1.values();
        assert_eq!(attrs, values);
    }
    #[test]
    fn test_composite_04() {
        let mut attrs = HashMap::new();
        attrs.insert((*COMMON_NAME).clone(), "AU".to_owned());
        attrs.insert((*ORGANIZATION_NAME).clone(), "The Legion of the Bouncy Castle".to_owned());
        attrs.insert((*LOCALITY_NAME).clone(), "Melbourne".to_owned());
        attrs.insert((*STREET).clone(), "Victoria".to_owned());
        attrs.insert((*PKCS9_AT_EMAIL_ADDRESS).clone(), "feedback-crypto@bouncycastle.org".to_owned());

        let mut order1 = Vec::new();
        order1.push((*COMMON_NAME).clone());
        order1.push((*ORGANIZATION_NAME).clone());
        order1.push((*LOCALITY_NAME).clone());
        order1.push((*STREET).clone());
        order1.push((*PKCS9_AT_EMAIL_ADDRESS).clone());

        let mut order2 = Vec::new();
        order2.push((*STREET).clone());
        order2.push((*STREET).clone());
        order2.push((*LOCALITY_NAME).clone());
        order2.push((*ORGANIZATION_NAME).clone());
        order2.push((*COMMON_NAME).clone());

        let name1 = X509Name::with_ordering_attributes(order1.clone(), attrs.clone()).unwrap();
        let name2 = X509Name::with_ordering_attributes(order2.clone(), attrs.clone()).unwrap();
        assert!(!name1.equivalent(&name2));

        let mut order2 = Vec::new();
        order2.push((*STREET).clone());
        order2.push((*LOCALITY_NAME).clone());
        order2.push((*ORGANIZATION_NAME).clone());
        order2.push((*COMMON_NAME).clone());
        let name2 = X509Name::with_ordering_attributes(order2.clone(), attrs.clone()).unwrap();
        assert!(!name1.equivalent(&name2));
    }
    // TODO: Add more tests for X509Name

    fn do_test_encoding_printable_string(oid: &Asn1ObjectIdentifier, value: &str) {
        let converted = create_entry_value(oid, value).unwrap();
        assert!(converted.is_printable_string());
    }
    fn do_test_encoding_ia5_string(oid: &Asn1ObjectIdentifier, value: &str) {
        let converted = create_entry_value(oid, value).unwrap();
        assert!(converted.is_ia5_string());
    }
    fn do_test_encoding_generalized_time(oid: &Asn1ObjectIdentifier, value: &str) {
        let converted = create_entry_value(oid, value).unwrap();
        assert!(converted.is_generalized_time());

        let converted = create_entry_value_from_string(oid, value).unwrap();
        assert!(converted.is_generalized_time());
    }
    fn create_entry_value(oid: &Asn1ObjectIdentifier, value: &str) -> Result<Asn1Object> {
        let mut attrs = HashMap::new();
        attrs.insert(oid.clone(), value.to_owned());

        let mut ord = Vec::new();
        ord.push(oid.clone());

        let name = X509Name::with_ordering_attributes(ord, attrs)?;

        let asn1_object: Asn1Object = name.into();
        let sequence: Asn1Sequence = asn1_object.try_into()?;
        let mut iter = sequence.into_iter();
        let set: Asn1Set = iter.next().unwrap().try_into()?;
        let mut iter = set.into_iter();
        let sequence: Asn1Sequence = iter.next().unwrap().try_into()?;
        let mut iter = sequence.into_iter();
        let _ = iter.next().unwrap(); // type
        let value_object = iter.next().unwrap();
        Ok(value_object)
    }
    fn create_entry_value_from_string(oid: &Asn1ObjectIdentifier, value: &str) -> Result<Asn1Object> {
        let mut attrs = HashMap::new();
        attrs.insert(oid.clone(), value.to_owned());

        let mut ord = Vec::new();
        ord.push(oid.clone());

        let name = X509Name::with_ordering_attributes(ord, attrs)?;
        let name = X509Name::with_str(&name.to_string())?;

        let asn1_object: Asn1Object = name.into();
        let sequence: Asn1Sequence = asn1_object.try_into()?;
        let mut iter = sequence.into_iter();
        let set: Asn1Set = iter.next().unwrap().try_into()?;
        let mut iter = set.into_iter();
        let sequence: Asn1Sequence = iter.next().unwrap().try_into()?;
        let mut iter = sequence.into_iter();
        let _ = iter.next().unwrap(); // type
        let value_object = iter.next().unwrap();
        Ok(value_object)
    }
}
