use crate::asn1::EncodingType::Ber;
use crate::asn1::asn1_utilities::try_from_choice_tagged;
use crate::asn1::pkcs::pkcs_object_identifiers::*;
use crate::asn1::try_from_tagged::TryFromTagged;
use crate::asn1::x500::style::ietf_utilities::{escape_dn_string, unescape};
use crate::asn1::x509::x509_object_identifiers::*;
use crate::asn1::x509::{X509DefaultEntryConverter, X509NameEntryConverter, X509NameTokenizer};
use crate::asn1::{
    Asn1Encodable, Asn1Object, Asn1ObjectIdentifier, Asn1Sequence, Asn1Set, Asn1TaggedObject,
    EncodingType,
};
use crate::util::encoders::hex::{to_decode_with_str, to_hex_string};
use crate::{BcError, GLOBAL, Result};
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::io::Write;
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
    fn new(
        ordering: Vec<Asn1ObjectIdentifier>,
        values: Vec<String>,
        added: Vec<bool>,
        converter: ConverterType,
    ) -> Self {
        X509Name {
            ordering,
            values,
            added,
            converter,
        }
    }
    pub fn with_str(dir_name: &str) -> Result<Self> {
        Self::with_reverse_lookup_str(
            (*GLOBAL).x509_name_default_reverse(),
            &DEFAULT_LOOKUP,
            dir_name,
        )
    }
    pub fn with_ordering_attributes(
        ordering: Vec<Asn1ObjectIdentifier>,
        attributes: Symbols,
    ) -> Result<Self> {
        Self::with_ordering_attributes_converter(
            ordering,
            attributes,
            Rc::new(X509DefaultEntryConverter),
        )
    }
    pub fn with_ordering_attributes_converter(
        ordering: Vec<Asn1ObjectIdentifier>,
        attributes: Symbols,
        converter: ConverterType,
    ) -> Result<Self> {
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
    pub fn with_reverse_str(reverse: bool, dir_name: &str) -> Result<Self> {
        Self::with_reverse_lookup_str(reverse, &DEFAULT_LOOKUP, dir_name)
    }
    pub fn with_reverse_lookup_str(
        reverse: bool,
        lookup: &LookupType,
        dir_name: &str,
    ) -> Result<Self> {
        Self::with_reverse_lookup_str_converter(
            reverse,
            lookup,
            dir_name,
            Rc::new(X509DefaultEntryConverter),
        )
    }
    pub fn with_reverse_lookup_str_converter(
        reverse: bool,
        lookup: &LookupType,
        dir_name: &str,
        converter: ConverterType,
    ) -> Result<Self> {
        let mut name_tokenizer = X509NameTokenizer::with_str(dir_name);

        let mut result = X509Name::new(Vec::new(), Vec::new(), Vec::new(), converter.clone());

        while let Some(rdn) = name_tokenizer.next_token()? {
            let mut rdn_tokenizer = X509NameTokenizer::with_str_and_separator(&rdn, '+')?;
            let token = rdn_tokenizer
                .next_token()?
                .ok_or(BcError::with_invalid_argument(
                    "badly formatted directory string",
                ))?;
            result.add_attribute(lookup, &token, false)?;
            while let Some(token) = rdn_tokenizer.next_token()? {
                result.add_attribute(lookup, &token, true)?;
            }
        }

        if reverse {
            let mut o = Vec::new();
            let mut v = Vec::new();
            let mut a = Vec::new();
            let mut count: isize = 1;

            for i in 0..result.ordering.len() {
                count &= if result.added[i] { -1 } else { 0 };
                o.insert(count as usize, result.ordering[i].clone());
                v.insert(count as usize, result.values[i].clone());
                a.insert(count as usize, result.added[i].clone());
                count += 1;
            }
            result = X509Name::new(o, v, a, converter.clone());
        }

        Ok(result)
    }
    pub fn with_ordering_values(
        ordering: Vec<Asn1ObjectIdentifier>,
        values: Vec<String>,
    ) -> Result<Self> {
        Self::with_ordering_values_converter(ordering, values, Rc::new(X509DefaultEntryConverter))
    }
    /// Takes two vectors one of the oids and the other of the values.
    /// The passed in converter will be used to convert the strings into their ASN.1 counterparts.
    pub fn with_ordering_values_converter(
        ordering: Vec<Asn1ObjectIdentifier>,
        values: Vec<String>,
        converter: ConverterType,
    ) -> Result<Self> {
        if ordering.len() != values.len() {
            return Err(BcError::with_invalid_argument(
                "'oids' must be same length as 'values'.",
            ));
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
                    return Err(BcError::with_invalid_format(
                        "badly sized AttributeTypeAndValue",
                    ));
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
                    values.push(format!(
                        "#{}",
                        to_hex_string(&(value_object.get_encoded(Ber)?))
                    ));
                }
                added.push(!first);
                if first {
                    first = false;
                }
            }
        }
        Ok(X509Name::new(
            ordering,
            values,
            added,
            Rc::new(X509DefaultEntryConverter),
        ))
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
    fn add_attribute(
        &mut self,
        lookup: &HashMap<String, Asn1ObjectIdentifier>,
        token: &str,
        added: bool,
    ) -> Result<()> {
        let mut tokenizer = X509NameTokenizer::with_str_and_separator(token, '=')?;

        let type_token = tokenizer
            .next_token()?
            .ok_or(BcError::with_invalid_argument(
                "badly formatted directory string",
            ))?;
        let value_token = tokenizer
            .next_token()?
            .ok_or(BcError::with_invalid_argument(
                "badly formatted directory string",
            ))?;

        let oid = Self::decode_oid(type_token.trim(), lookup)?;
        let value = unescape(&value_token);

        self.ordering.push(oid);
        self.values.push(value);
        self.added.push(added);

        if tokenizer.has_more_tokens() {
            return Err(BcError::with_invalid_argument(
                "badly formatted directory string - too many tokens",
            ));
        }

        Ok(())
    }
    fn decode_oid(
        name: &str,
        lookup: &HashMap<String, Asn1ObjectIdentifier>,
    ) -> Result<Asn1ObjectIdentifier> {
        if name.starts_with("OID.") || name.starts_with("oid.") {
            return Ok(Asn1ObjectIdentifier::with_str(&name[4..])?);
        }

        if let Some(r) = lookup.get(&name.to_lowercase()) {
            return Ok(r.clone());
        }

        if let Some(r) = Asn1ObjectIdentifier::try_parse(name) {
            return Ok(r);
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
                    v = obj_str
                        .to_asn1_string()
                        .unwrap()
                        .to_lowercase()
                        .trim()
                        .to_string();
                }
            }
        }
        v.to_string()
    }
    fn decode_object(s: &str) -> Result<Asn1Object> {
        let buffer = to_decode_with_str(&s[1..])?;
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
    pub fn values_by_oid(&self, oid: &Asn1ObjectIdentifier) -> Vec<String> {
        let mut result = Vec::new();
        for (i, o) in self.ordering.iter().enumerate() {
            if o == oid {
                let mut value = self.values.get(i).unwrap().to_string();
                if value.starts_with("\\#") {
                    value = value[1..].to_string();
                }
                result.push(value)
            }
        }
        result
    }
}
impl fmt::Display for X509Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.to_string_with_symbols(false, &DEFAULT_SYMBOLS)
        )
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
impl Asn1Encodable for X509Name {
    fn encode_to(&self, writer: &mut dyn Write, encoding_type: EncodingType) -> Result<usize> {
        let asn1_object: Asn1Object = self.clone().into();
        asn1_object.encode_to(writer, encoding_type)
    }
}
fn append_value(
    buffer: &mut String,
    oid_symbols: &Symbols,
    oid: &Asn1ObjectIdentifier,
    value: &str,
) {
    buffer.push_str(oid_symbols.get(oid).unwrap_or(oid.id()));
    buffer.push('=');
    buffer.push_str(&escape_dn_string(value));
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
    symbols.insert(
        PKCS9_AT_UNSTRUCTURED_ADDRESS.clone(),
        "unstructuredAddress".to_owned(),
    );
    symbols.insert(
        PKCS9_AT_UNSTRUCTURED_NAME.clone(),
        "unstructuredName".to_owned(),
    );
    symbols.insert(UNIQUE_IDENTIFIER.clone(), "UniqueIdentifier".to_owned());
    symbols.insert(DN_QUALIFIER.clone(), "DN".to_owned());
    symbols.insert(PSEUDONYM.clone(), "Pseudonym".to_owned());
    symbols.insert(POSTAL_ADDRESS.clone(), "PostalAddress".to_owned());
    symbols.insert(NAME_AT_BIRTH.clone(), "NameAtBirth".to_owned());
    symbols.insert(
        COUNTRY_OF_CITIZENSHIP.clone(),
        "CountryOfCitizenship".to_owned(),
    );
    symbols.insert(
        COUNTRY_OF_RESIDENCE.clone(),
        "CountryOfResidence".to_owned(),
    );
    symbols.insert(GENDER.clone(), "Gender".to_owned());
    symbols.insert(PLACE_OF_BIRTH.clone(), "PlaceOfBirth".to_owned());
    symbols.insert(DATE_OF_BIRTH.clone(), "DateOfBirth".to_owned());
    symbols.insert(POSTAL_CODE.clone(), "PostalCode".to_owned());
    symbols.insert(BUSINESS_CATEGORY.clone(), "BusinessCategory".to_owned());
    symbols.insert(TELEPHONE_NUMBER.clone(), "TelephoneNumber".to_owned());
    symbols.insert(NAME.clone(), "Name".to_owned());
    symbols.insert(
        ORGANIZATION_IDENTIFIER.clone(),
        "organizationIdentifier".to_owned(),
    );
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
    lookup.insert("cn".to_owned(), COMMON_NAME.clone());
    lookup.insert("l".to_owned(), LOCALITY_NAME.clone());
    lookup.insert("st".to_owned(), STATE_OR_PROVINCE_NAME.clone());
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
    lookup.insert(
        "unstructuredaddress".to_owned(),
        PKCS9_AT_UNSTRUCTURED_ADDRESS.clone(),
    );
    lookup.insert(
        "unstructuredname".to_owned(),
        PKCS9_AT_UNSTRUCTURED_NAME.clone(),
    );
    lookup.insert("uniqueidentifier".to_owned(), UNIQUE_IDENTIFIER.clone());
    lookup.insert("dn".to_owned(), DN_QUALIFIER.clone());
    lookup.insert("pseudonym".to_owned(), PSEUDONYM.clone());
    lookup.insert("postaladdress".to_owned(), POSTAL_ADDRESS.clone());
    lookup.insert("nameatbirth".to_owned(), NAME_AT_BIRTH.clone());
    lookup.insert(
        "countryofcitizenship".to_owned(),
        COUNTRY_OF_CITIZENSHIP.clone(),
    );
    lookup.insert(
        "countryofresidence".to_owned(),
        COUNTRY_OF_RESIDENCE.clone(),
    );
    lookup.insert("gender".to_owned(), GENDER.clone());
    lookup.insert("placeofbirth".to_owned(), PLACE_OF_BIRTH.clone());
    lookup.insert("dateofbirth".to_owned(), DATE_OF_BIRTH.clone());
    lookup.insert("postalcode".to_owned(), POSTAL_CODE.clone());
    lookup.insert("businesscategory".to_owned(), BUSINESS_CATEGORY.clone());
    lookup.insert("telephonenumber".to_owned(), TELEPHONE_NUMBER.clone());
    lookup.insert("name".to_owned(), ID_AT_NAME.clone());
    lookup.insert(
        "organizationidentifier".to_owned(),
        ORGANIZATION_IDENTIFIER.clone(),
    );
    lookup.insert("jurisdictioncountry".to_owned(), JURISDICTION_C.clone());
    lookup.insert("jurisdictionstate".to_owned(), JURISDICTION_ST.clone());
    lookup.insert("jurisdictionlocality".to_owned(), JURISDICTION_L.clone());
    lookup
});

#[cfg(test)]
mod tests {
    use crate::Result;
    use crate::asn1::EncodingType::{Ber, Der};
    use crate::asn1::pkcs::pkcs_object_identifiers::PKCS9_AT_EMAIL_ADDRESS;
    use crate::asn1::x500::style::ietf_utilities::asn1_object_to_string;
    use crate::asn1::x509::x509_object_identifiers::*;
    use crate::asn1::x509::{
        X509DefaultEntryConverter, X509Name, X509NameEntryConverter, x509_name,
    };
    use crate::asn1::{
        Asn1Encodable, Asn1EncodableVector, Asn1Object, Asn1ObjectIdentifier, Asn1Sequence,
        Asn1Set, Asn1Utf8String,
    };
    use crate::util::encoders::hex::{to_decode_with_str, to_hex_string};
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
        do_test_encoding_printable_string(&(*COUNTRY_NAME), "AU");
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
        attrs.insert(
            (*ORGANIZATION_NAME).clone(),
            "The Legion of the Bouncy Castle".to_owned(),
        );
        attrs.insert((*LOCALITY_NAME).clone(), "Melbourne".to_owned());
        attrs.insert((*STREET).clone(), "Victoria".to_owned());
        attrs.insert(
            (*PKCS9_AT_EMAIL_ADDRESS).clone(),
            "feedback-crypto@bouncycastle.org".to_owned(),
        );

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
        attrs.insert(
            (*ORGANIZATION_NAME).clone(),
            "The Legion of the Bouncy Castle".to_owned(),
        );
        attrs.insert((*LOCALITY_NAME).clone(), "Melbourne".to_owned());
        attrs.insert((*STREET).clone(), "Victoria".to_owned());
        attrs.insert(
            (*PKCS9_AT_EMAIL_ADDRESS).clone(),
            "feedback-crypto@bouncycastle.org".to_owned(),
        );

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
        attrs.insert(
            (*ORGANIZATION_NAME).clone(),
            "The Legion of the Bouncy Castle".to_owned(),
        );
        attrs.insert((*LOCALITY_NAME).clone(), "Melbourne".to_owned());
        attrs.insert((*STREET).clone(), "Victoria".to_owned());
        attrs.insert(
            (*PKCS9_AT_EMAIL_ADDRESS).clone(),
            "feedback-crypto@bouncycastle.org".to_owned(),
        );

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
    #[test]
    fn test_composite_05() {
        let enc = to_decode_with_str("305e310b300906035504061302415531283026060355040a0c1f546865204c6567696f6e206f662074686520426f756e637920436173746c653125301006035504070c094d656c626f75726e653011060355040b0c0a4173636f742056616c65").unwrap();
        let n: X509Name = Asn1Object::with_bytes(&enc).unwrap().try_into().unwrap();
        assert_eq!(
            "C=AU,O=The Legion of the Bouncy Castle,L=Melbourne+OU=Ascot Vale",
            n.to_string()
        );

        let symbols = &(*x509_name::DEFAULT_SYMBOLS);
        assert_eq!(
            "L=Melbourne+OU=Ascot Vale,O=The Legion of the Bouncy Castle,C=AU",
            n.to_string_with_symbols(true, symbols)
        );

        let n = X509Name::with_reverse_str(
            true,
            "L=Melbourne+OU=Ascot Vale,O=The Legion of the Bouncy Castle,C=AU",
        )
        .unwrap();
        assert_eq!(
            "C=AU,O=The Legion of the Bouncy Castle,L=Melbourne+OU=Ascot Vale",
            n.to_string()
        );

        let n = X509Name::with_str(
            "C=AU, O=The Legion of the Bouncy Castle, L=Melbourne + OU=Ascot Vale",
        )
        .unwrap();
        let enc2 = n.get_encoded(Ber).unwrap();
        assert_eq!(enc, enc2);

        let n = X509Name::with_str("C=CH,O=,OU=dummy,CN=mail@dummy.com").unwrap();
        let buffer = n.get_encoded(Ber).unwrap();
        let _: X509Name = Asn1Object::with_bytes(&buffer).unwrap().try_into().unwrap();
    }
    #[test]
    fn test_get_values() {
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
        let values = name1.values_by_oid(&(*ORGANIZATION_NAME));
        assert_eq!(values.len(), 1);
        assert_eq!(values[0], "The Legion of the Bouncy Castle");

        let values = name1.values_by_oid(&(*LOCALITY_NAME));
        assert_eq!(values.len(), 1);
        assert_eq!(values[0], "Melbourne");
    }
    #[test]
    fn test_general_subjects_01() {
        const SUBJECT: [&str; 12] = [
            "C=AU,ST=Victoria,L=South Melbourne,O=Connect 4 Pty Ltd,OU=Webserver Team,CN=www2.connect4.com.au,E=webmaster@connect4.com.au",
            "C=AU,ST=Victoria,L=South Melbourne,O=Connect 4 Pty Ltd,OU=Certificate Authority,CN=Connect 4 CA,E=webmaster@connect4.com.au",
            "C=AU,ST=QLD,CN=SSLeay/rsa test cert",
            "C=US,O=National Aeronautics and Space Administration,SERIALNUMBER=16+CN=Steve Schoch",
            "E=cooke@issl.atl.hp.com,C=US,OU=Hewlett Packard Company (ISSL),CN=Paul A. Cooke",
            "O=Sun Microsystems Inc,CN=store.sun.com",
            "unstructuredAddress=192.168.1.33,unstructuredName=pixfirewall.ciscopix.com,CN=pixfirewall.ciscopix.com",
            "CN=*.canal-plus.com,OU=Provided by TBS INTERNET https://www.tbs-certificats.com/,OU=\\ CANAL \\+,O=CANAL\\+DISTRIBUTION,L=issy les moulineaux,ST=Hauts de Seine,C=FR",
            "O=Bouncy Castle,CN=www.bouncycastle.org\\ ",
            "O=Bouncy Castle,CN=c:\\\\fred\\\\bob",
            concat!(
                "C=0,O=1,OU=2,T=3,CN=4,SERIALNUMBER=5,STREET=6,SERIALNUMBER=7,L=8,ST=9,SURNAME=10,GIVENNAME=11,INITIALS=12,",
                "GENERATION=13,UniqueIdentifier=14,BusinessCategory=15,PostalCode=16,DN=17,Pseudonym=18,PlaceOfBirth=19,",
                "Gender=20,CountryOfCitizenship=21,CountryOfResidence=22,NameAtBirth=23,PostalAddress=24,2.5.4.54=25,",
                "TelephoneNumber=26,Name=27,E=28,unstructuredName=29,unstructuredAddress=30,E=31,DC=32,UID=33"
            ),
            "C=DE,L=Berlin,O=Wohnungsbaugenossenschaft \\\"Humboldt-Universit酹\\\" eG,CN=transfer.wbg-hub.de",
        ];

        for s in SUBJECT {
            let name = X509Name::with_str(s).unwrap();
            let buffer = name.get_encoded(Ber).unwrap();
            let name: X509Name = Asn1Object::with_bytes(&buffer).unwrap().try_into().unwrap();
            let decode_subject = name.to_string();
            assert_eq!(s, decode_subject);
        }
    }
    #[test]
    fn test_general_subjects_02() {
        const HEX_SUBJECT: [&str; 4] = [
            "CN=\\20Test\\20X,O=\\20Test,C=GB",
            "CN=\\ Test X,O=\\ Test,C=GB",
            "CN=\\20Test\\20X\\20,O=\\20Test,C=GB",
            "CN=\\ Test X\\ ,O=\\ Test,C=GB",
        ];

        for chunk in HEX_SUBJECT.chunks(2) {
            let subject = chunk[0];
            let expected = chunk[1];

            let name = X509Name::with_str(subject).unwrap();
            let decoded_name: X509Name = Asn1Object::with_bytes(&name.get_encoded(Ber).unwrap())
                .unwrap()
                .try_into()
                .unwrap();
            let decode_subject = decoded_name.to_string();
            assert_eq!(expected, decode_subject);
        }
    }
    #[test]
    fn test_sort_01() {
        let unsorted = X509Name::with_str("SERIALNUMBER=BBB + CN=AA").unwrap();
        let name: X509Name = Asn1Object::with_bytes(&unsorted.get_encoded(Der).unwrap())
            .unwrap()
            .try_into()
            .unwrap();
        assert_eq!("CN=AA+SERIALNUMBER=BBB", name.to_string());

        let unsorted = X509Name::with_str("CN=AA + SERIALNUMBER=BBB").unwrap();
        let name: X509Name = Asn1Object::with_bytes(&unsorted.get_encoded(Der).unwrap())
            .unwrap()
            .try_into()
            .unwrap();
        assert_eq!("CN=AA+SERIALNUMBER=BBB", name.to_string());

        let unsorted = X509Name::with_str("SERIALNUMBER=B + CN=AA").unwrap();
        let name: X509Name = Asn1Object::with_bytes(&unsorted.get_encoded(Der).unwrap())
            .unwrap()
            .try_into()
            .unwrap();
        assert_eq!("SERIALNUMBER=B+CN=AA", name.to_string());

        let unsorted = X509Name::with_str("CN=AA + SERIALNUMBER=B").unwrap();
        let name: X509Name = Asn1Object::with_bytes(&unsorted.get_encoded(Der).unwrap())
            .unwrap()
            .try_into()
            .unwrap();
        assert_eq!("SERIALNUMBER=B+CN=AA", name.to_string());
    }
    #[test]
    fn test_equality_01() {
        test_equality(
            &X509Name::with_str("CN=The     Legion").unwrap(),
            &X509Name::with_str("CN=The Legion").unwrap(),
        );
        test_equality(
            &X509Name::with_str("CN=   The Legion").unwrap(),
            &X509Name::with_str("CN=The Legion").unwrap(),
        );
        test_equality(
            &X509Name::with_str("CN=The Legion   ").unwrap(),
            &X509Name::with_str("CN=The Legion").unwrap(),
        );
        test_equality(
            &X509Name::with_str("CN=  The     Legion ").unwrap(),
            &X509Name::with_str("CN=The Legion").unwrap(),
        );
        test_equality(
            &X509Name::with_str("CN=  the     legion ").unwrap(),
            &X509Name::with_str("CN=The Legion").unwrap(),
        );
    }
    #[test]
    fn test_equality_02() {
        let n1 = X509Name::with_str("SERIALNUMBER=8,O=ABC,CN=ABC Class 3 CA,C=LT").unwrap();
        let n2 = X509Name::with_str("2.5.4.5=8,O=ABC,CN=ABC Class 3 CA,C=LT").unwrap();
        let n3 = X509Name::with_str("2.5.4.5=#130138,O=ABC,CN=ABC Class 3 CA,C=LT").unwrap();
        test_equality(&n1, &n2);
        test_equality(&n2, &n3);
        test_equality(&n3, &n1);
    }
    #[test]
    fn test_equality_03() {
        let n1 = X509Name::with_reverse_str(
            true,
            "2.5.4.5=#130138,CN=SSC Class 3 CA,O=UAB Skaitmeninio sertifikavimo centras,C=LT",
        )
        .unwrap();
        let n2 = X509Name::with_reverse_str(
            true,
            "SERIALNUMBER=#130138,CN=SSC Class 3 CA,O=UAB Skaitmeninio sertifikavimo centras,C=LT",
        )
        .unwrap();
        let n3: X509Name = Asn1Object::with_bytes(&to_decode_with_str("3063310b3009060355040613024c54312f302d060355040a132655414220536b6169746d656e696e696f20736572746966696b6176696d6f2063656e74726173311730150603550403130e53534320436c6173732033204341310a30080603550405130138").unwrap()).unwrap().try_into().unwrap();
        test_equality(&n1, &n2);
        test_equality(&n2, &n3);
        test_equality(&n3, &n1);
    }
    #[test]
    fn test_equality_04() {
        let n1 = X509Name::with_str("SERIALNUMBER=8,O=XX,CN=ABC Class 3 CA,C=LT").unwrap();
        let n2 = X509Name::with_str("2.5.4.5=8,O=,CN=ABC Class 3 CA,C=LT").unwrap();
        assert!(!n1.equivalent(&n2));
    }
    #[test]
    fn test_equality_05() {
        let n1 = X509Name::with_str("").unwrap();
        let n2 = X509Name::with_str("").unwrap();
        test_equality(&n1, &n2);
    }
    #[test]
    fn test_inequality_to_sequence_01() {
        let name1: Asn1Object = X509Name::with_str("CN=The Legion").unwrap().into();
        let sequence: Asn1Object = Asn1Sequence::new(Vec::new()).into();
        assert_ne!(name1, sequence)
    }
    #[test]
    fn test_inequality_to_sequence_02() {
        let name1: Asn1Object = X509Name::with_str("CN=The Legion").unwrap().into();
        let sequence: Asn1Object = Asn1Sequence::new(vec![Asn1Set::new(Vec::new()).into()]).into();
        assert_ne!(name1, sequence)
    }
    #[test]
    fn test_inequality_to_sequence_03() {
        let name1: Asn1Object = X509Name::with_str("CN=The Legion").unwrap().into();
        let v = Asn1EncodableVector::new(vec![
            Asn1ObjectIdentifier::with_str("1.1").unwrap().into(),
            Asn1ObjectIdentifier::with_str("1.1").unwrap().into(),
        ]);
        let sequence: Asn1Object = Asn1Sequence::new(vec![
            Asn1Set::new(vec![Asn1Set::from_vector(v).unwrap().into()]).into(),
        ])
        .into();
        assert_ne!(name1, sequence);

        let sequence: Asn1Sequence = sequence.try_into().unwrap();
        assert!(X509Name::from_sequence(sequence).is_err());
    }
    #[test]
    fn test_inequality_to_sequence_04() {
        let name1: Asn1Object = X509Name::with_str("CN=The Legion").unwrap().into();
        let sequence: Asn1Object = Asn1Sequence::new(vec![
            Asn1Set::new(vec![Asn1Sequence::empty().into()]).into(),
        ])
        .into();
        assert_ne!(name1, sequence)
    }
    #[test]
    fn test_escaped_01() {
        let test_string = Asn1Utf8String::with_str("The Legion of the Bouncy Castle");
        let encoded_bytes = test_string.get_encoded(Ber).unwrap();
        let hex_encoded_string = format!("#{}", to_hex_string(&encoded_bytes));

        let converter = X509DefaultEntryConverter;
        let converted: Asn1Utf8String = converter
            .get_converted_value(&(*LOCALITY_NAME), &hex_encoded_string)
            .unwrap()
            .try_into()
            .unwrap();
        assert_eq!(test_string, converted);
    }
    #[test]
    fn test_escaped_02() {
        let test_string = Asn1Utf8String::with_str("The Legion of the Bouncy Castle");
        let encoded_bytes = test_string.get_encoded(Ber).unwrap();
        let hex_encoded_string = format!("#{}", to_hex_string(&encoded_bytes));

        let converter = X509DefaultEntryConverter;
        let converted: Asn1Utf8String = converter
            .get_converted_value(&(*LOCALITY_NAME), &format!("\\{}", &hex_encoded_string))
            .unwrap()
            .try_into()
            .unwrap();
        assert_eq!(converted, Asn1Utf8String::with_str(&hex_encoded_string));
    }
    #[test]
    fn test_weird_value_01() {
        let n = X509Name::with_str("CN=\\#nothex#string").unwrap();
        assert_eq!("CN=\\#nothex#string", n.to_string());

        let values = n.values_by_oid(&(*COMMON_NAME));
        assert_eq!(values.len(), 1);
        assert_eq!(values[0], "#nothex#string");
    }
    #[test]
    fn test_weird_value_02() {
        let n = X509Name::with_str("CN=\"a+b\"").unwrap();
        let values = n.values_by_oid(&(*COMMON_NAME));
        assert_eq!(values.len(), 1);
        assert_eq!(values[0], "a+b");
    }
    #[test]
    fn test_weird_value_03() {
        let n = X509Name::with_str("CN=a\\+b").unwrap();
        let values = n.values_by_oid(&(*COMMON_NAME));
        assert_eq!(values.len(), 1);
        assert_eq!(values[0], "a+b");
        assert_eq!("CN=a\\+b", n.to_string());
    }
    #[test]
    fn test_weird_value_04() {
        let n = X509Name::with_str("CN=a\\=b").unwrap();
        let values = n.values_by_oid(&(*COMMON_NAME));
        assert_eq!(values.len(), 1);
        assert_eq!(values[0], "a=b");
        assert_eq!("CN=a\\=b", n.to_string());
    }
    #[test]
    fn test_weird_value_05() {
        let n = X509Name::with_str("TELEPHONENUMBER=\"+61999999999\"").unwrap();
        let values = n.values_by_oid(&(*TELEPHONE_NUMBER));
        assert_eq!(values.len(), 1);
        assert_eq!(values[0], "+61999999999");
    }
    #[test]
    fn test_weird_value_06() {
        let n = X509Name::with_str("TELEPHONENUMBER=\\+61999999999").unwrap();
        let values = n.values_by_oid(&(*TELEPHONE_NUMBER));
        assert_eq!(values.len(), 1);
        assert_eq!(values[0], "+61999999999");
    }
    #[test]
    fn test_weird_value_07() {
        let n = X509Name::with_str("TELEPHONENUMBER=\\+61999999999").unwrap();
        let values = n.values_by_oid(&(*TELEPHONE_NUMBER));
        assert_eq!(values.len(), 1);
        assert_eq!(values[0], "+61999999999");
    }
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
    fn create_entry_value_from_string(
        oid: &Asn1ObjectIdentifier,
        value: &str,
    ) -> Result<Asn1Object> {
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
    fn test_equality(n1: &X509Name, n2: &X509Name) {
        assert!(n1.equivalent(n2));
        assert!(n1.equivalent_in_order(n2, true));
    }
}
