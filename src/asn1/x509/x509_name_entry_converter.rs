use crate::Result;
use crate::asn1::{Asn1Object, Asn1ObjectIdentifier};
use crate::util::encoders::hex::to_decode_with_str;

/// It turns out that the number of standard ways the fields in a DN should be
/// encoded into their ASN.1 counterparts is rapidly approaching the
/// number of machines on the internet. By default, the X509Name class
/// will produce UTF8Strings in line with the current recommendations (RFC 3280).
pub trait X509NameEntryConverter {
    /// Convert the passed in string value into the appropriate ASN.1 encoded object.
    ///
    /// # Arguments
    /// * `oid` - the oid associated with the value in the DN.
    /// * `value` - the value of the particular DN component.
    ///
    /// # Returns
    /// * `Result<Asn1Object>` - the ASN.1 equivalent for the value.
    fn get_converted_value(&self, oid: &Asn1ObjectIdentifier, value: &str) -> Result<Asn1Object>;
}

/// Convert an inline encoded hex string rendition of an ASN.1
/// object back into its corresponding ASN.1 object.
///
/// # Arguments
/// * `value` - A string that represents a hex-encoded ASN.1 object.
///
/// # Returns
/// * `Result<Asn1Object>` - the decoded object
pub(crate) fn convert_hex_encoded(value: &str) -> Result<Asn1Object> {
    let buffer = to_decode_with_str(value)?;
    Asn1Object::with_bytes(&buffer)
}
