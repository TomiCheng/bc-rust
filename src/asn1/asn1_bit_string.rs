use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_tags::{BIT_STRING, UNIVERSAL};
use crate::asn1::asn1_universal_type::Asn1UniversalType;
use crate::asn1::primitive_encoding::PrimitiveEncoding;
use crate::asn1::primitive_encoding_suffixed::PrimitiveEncodingSuffixed;
use crate::asn1::try_from_tagged::TryFromTagged;
use crate::asn1::{Asn1Object, Asn1TaggedObject, EncodingType};
use crate::{BcError, Result};
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Asn1BitString {
    contents: Vec<u8>,
    pad_bits: u8,
}
impl Asn1BitString {
    pub fn with_pad_bits(contents: &[u8], pad_bits: u8) -> Result<Self> {
        if pad_bits > 7 {
            return Err(BcError::with_invalid_argument("Pad bits must be between 0 and 7"));
        }
        if contents.is_empty() && pad_bits > 0 {
            return Err(BcError::with_invalid_argument("if 'content' is empty, 'pad_bits' must be 0"));
        }
        Ok(Self {
            contents: contents.to_vec(),
            pad_bits,
        })
    }
    pub fn with_named_bits(value: u32) -> Self {
        if value == 0 {
            return Asn1BitString {
                contents: vec![],
                pad_bits: 0,
            };
        }
        let bits = 32 - u32::leading_zeros(value) as usize;
        let bytes = (bits + 7) / 8;
        debug_assert!(0 < bytes && bytes <= 4);

        let mut contents = vec![0u8; bytes];
        let mut named_bits = value;
        for i in 0..(bytes - 1) {
            contents[i] = named_bits as u8;
            named_bits >>= 8;
        }
        debug_assert!((named_bits & 0xFF) != 0);
        contents[bytes - 1] = named_bits as u8;

        let mut pad_bits = 0;
        while named_bits & (1 << pad_bits) == 0 {
            pad_bits += 1;
        }

        debug_assert!(pad_bits < 8);
        Asn1BitString { contents, pad_bits }
    }
    pub fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        if contents.is_empty() {
            return Err(BcError::with_invalid_argument("truncated BIT STRING detected"));
        }

        let length = contents.len();
        let pad_bits = contents[0];
        let contents = contents[1..].to_vec();

        if pad_bits > 0 {
            if pad_bits > 7 || length < 2 {
                return Err(BcError::with_invalid_argument("invalid pad bits detected"));
            }
        }
        Ok(Asn1BitString { contents, pad_bits })
    }
    pub fn to_vec(&self) -> Vec<u8> {
        let mut result = self.contents.to_vec();
        if !self.contents.is_empty() {
            // DER requires pad bits be zero
            let last = result.len() - 1;
            result[last] &= 0xFF << self.pad_bits;
        }
        result
    }
    pub fn get_pad_bits(&self) -> u8 {
        self.pad_bits
    }
    pub fn get_contents(&self) -> &[u8] {
        &self.contents
    }
}
impl Asn1EncodingInternal for Asn1BitString {
    fn get_encoding(&self, encoding_type: EncodingType) -> Box<dyn Asn1Encoding> {
        let mut contents = vec![0u8; 1 + self.contents.len()];
        contents[0] = self.pad_bits;
        contents[1..].copy_from_slice(&self.contents);

        if encoding_type == EncodingType::Der && self.pad_bits != 0 {
            // let last = self.contents.len();
            let last_ber = self.contents[self.contents.len() - 1];
            let last_der = last_ber & (0xFF << self.pad_bits);
            if last_ber != last_der {
                contents[self.contents.len()] = last_der;
                return Box::new(PrimitiveEncodingSuffixed::new(UNIVERSAL, BIT_STRING, contents, last_der));
            }
        }
        Box::new(PrimitiveEncoding::new(UNIVERSAL, BIT_STRING, contents))
    }
}
impl TryFromTagged for Asn1BitString {
    fn try_from_tagged(tagged: Asn1TaggedObject, declared_explicit: bool) -> std::result::Result<Self, BcError>
    where
        Self: Sized,
    {
        tagged.try_from_base_universal(declared_explicit, Asn1BitStringMetadata)
    }
}
impl Hash for Asn1BitString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        todo!();
    }
}
struct Asn1BitStringMetadata;
impl Asn1UniversalType<Asn1BitString> for Asn1BitStringMetadata {
    fn checked_cast(&self, asn1_object: Asn1Object) -> Result<Asn1BitString> {
        asn1_object.try_into()
    }
}

#[cfg(test)]
mod tests {
    use crate::asn1::EncodingType::{Ber, Der};
    use crate::asn1::{Asn1BitString, Asn1Encodable, Asn1Object};
    use crate::util::encoders::hex::to_decode_with_str;

    #[test]
    fn test_zero_length_strings() {
        let s1 = Asn1BitString::with_pad_bits(&vec![], 0).unwrap();
        s1.to_vec();

        let buffer = s1.get_encoded(Ber).unwrap();
        assert_eq!(buffer, vec![0x03, 0x01, 0x00]);

        assert!(Asn1BitString::with_pad_bits(&vec![], 1).is_err());
        assert!(Asn1BitString::with_pad_bits(&vec![0], 8).is_err());

        let s2 = Asn1BitString::with_named_bits(0);
        let buffer2 = s2.get_encoded(Ber).unwrap();
        assert_eq!(buffer2, buffer);
    }
    #[test]
    fn test_with_pad_bits_fail() {
        assert!(Asn1BitString::with_pad_bits(&vec![], 1).is_err());
        assert!(Asn1BitString::with_pad_bits(&vec![0], 8).is_err());
    }
    #[test]
    fn test_random_pad_bits() {
        let test = to_decode_with_str("030206c0").unwrap();
        let test1 = to_decode_with_str("030206f0").unwrap();
        let test2 = to_decode_with_str("030206c1").unwrap();
        let test3 = to_decode_with_str("030206c7").unwrap();
        let test4 = to_decode_with_str("030206d1").unwrap();
        check_encoding(&test, &test1);
        check_encoding(&test, &test2);
        check_encoding(&test, &test3);
        check_encoding(&test, &test4);
    }
    fn check_encoding(der_data: &Vec<u8>, dl_data: &Vec<u8>) {
        let asn1_object = Asn1Object::with_bytes(dl_data).unwrap();

        assert_ne!(der_data, &asn1_object.get_encoded(Ber).unwrap());

        assert!(asn1_object.is_bit_string());
        let bit_string = asn1_object.as_bit_string().unwrap();
        let der_buffer = bit_string.get_encoded(Der).unwrap();
        assert_eq!(der_data, &der_buffer);
    }
}
