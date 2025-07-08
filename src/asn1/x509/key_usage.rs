use crate::asn1::{Asn1BitString, Asn1Object};
use crate::Result;

/// The KeyUsage object.
/// ```text
///  id-ce-keyUsage OBJECT IDENTIFIER ::=  { id-ce 15 }
///
///  KeyUsage ::= BIT STRING {
///       digitalSignature        (0),
///       nonRepudiation          (1),
///       keyEncipherment         (2),
///       dataEncipherment        (3),
///       keyAgreement            (4),
///       keyCertSign             (5),
///       cRLSign                 (6),
///       encipherOnly            (7),
///       decipherOnly            (8) }
/// ```
pub struct KeyUsage {
    content: Asn1BitString
}

impl KeyUsage {
    pub const DECIPHER_ONLY: u32 = 1 << 15;
    pub const DIGITAL_SIGNATURE: u32 = 1 << 7;
    pub const NON_REPUDIATION: u32 = 1 << 6;
    pub const KEY_ENCIPHERMENT: u32 = 1 << 5;
    pub const DATA_ENCIPHERMENT: u32 = 1 << 4;
    pub const KEY_AGREEMENT: u32 = 1 << 3;
    pub const KEY_CERT_SIGN: u32 = 1 << 2;
    pub const CRL_SIGN: u32 = 1 << 1;
    pub const ENCIPHER_ONLY: u32 = 1 << 0;
    fn new(content: Asn1BitString) -> Self {
        KeyUsage { content }
    }
    pub fn with_u32(value: u32) -> Self {
        let content = Asn1BitString::with_named_bits(value);
        KeyUsage { content }
    }
}
impl AsRef<Asn1BitString> for KeyUsage {
    fn as_ref(&self) -> &Asn1BitString {
        &self.content
    }
}
impl TryFrom<Asn1Object> for KeyUsage {
    type Error = crate::BcError;

    fn try_from(asn1_object: Asn1Object) -> Result<Self> {
        if let Asn1Object::BitString(content) = asn1_object {
            return Ok(KeyUsage::new(content));
        }
        todo!()
    }
}
#[cfg(test)]
mod tests {
    use crate::asn1::x509::KeyUsage;

    #[test]
    fn test() {
        let key_usage = KeyUsage::with_u32(KeyUsage::DIGITAL_SIGNATURE);
        assert!(!(key_usage.as_ref().get_contents()[0] != (KeyUsage::DIGITAL_SIGNATURE as u8) || key_usage.as_ref().get_pad_bits() != 7));
        let key_usage = KeyUsage::with_u32(KeyUsage::NON_REPUDIATION);
        assert!(!(key_usage.as_ref().get_contents()[0] != (KeyUsage::NON_REPUDIATION as u8) || key_usage.as_ref().get_pad_bits() != 6));
        let key_usage = KeyUsage::with_u32(KeyUsage::KEY_ENCIPHERMENT);
        assert!(!(key_usage.as_ref().get_contents()[0] != (KeyUsage::KEY_ENCIPHERMENT as u8) || key_usage.as_ref().get_pad_bits() != 5));
        let key_usage = KeyUsage::with_u32(KeyUsage::CRL_SIGN);
        assert!(!(key_usage.as_ref().get_contents()[0] != (KeyUsage::CRL_SIGN as u8) || key_usage.as_ref().get_pad_bits() != 1));
        let key_usage = KeyUsage::with_u32(KeyUsage::DECIPHER_ONLY);
        assert!(!(key_usage.as_ref().get_contents()[1] != ((KeyUsage::DECIPHER_ONLY >> 8) as u8) || key_usage.as_ref().get_pad_bits() != 7));
    }
}