use crate::Result;
use crate::asn1::pkcs::pkcs_object_identifiers::PKCS9_AT_EMAIL_ADDRESS;
use crate::asn1::x509::X509NameEntryConverter;
use crate::asn1::x509::x509_name_entry_converter::convert_hex_encoded;
use crate::asn1::x509::x509_object_identifiers::{COUNTRY_NAME, DATE_OF_BIRTH, DC, DN_QUALIFIER, SERIAL_NUMBER, TELEPHONE_NUMBER};
use crate::asn1::{Asn1GeneralizedTime, Asn1Ia5String, Asn1Object, Asn1ObjectIdentifier, Asn1PrintableString, Asn1Utf8String};

/// The default converter for X509 DN entries when going from their string value to ASN.1 strings.
pub struct X509DefaultEntryConverter;

impl X509NameEntryConverter for X509DefaultEntryConverter {
    fn get_converted_value(&self, oid: &Asn1ObjectIdentifier, value: &str) -> Result<Asn1Object> {
        let mut value = value;

        if value.chars().nth(0) == Some('#') {
            return convert_hex_encoded(&value[1..]);
        }
        if value.chars().nth(0) == Some('\\') {
            value = &value[1..];
        }

        if oid == &(*PKCS9_AT_EMAIL_ADDRESS) || oid == &(*DC) {
            return Ok(Asn1Ia5String::with_str(value)?.into());
        }

        if oid == &(*DATE_OF_BIRTH) {
            return Ok(Asn1GeneralizedTime::with_str(value)?.into());
        }

        if oid == &(*COUNTRY_NAME) || oid == &(*SERIAL_NUMBER) || oid == &(*DN_QUALIFIER) || oid == &(*TELEPHONE_NUMBER) {
            return Ok(Asn1PrintableString::with_str_validate(value, false)?.into());
        }

        Ok(Asn1Utf8String::with_str(value).into())
    }
}
