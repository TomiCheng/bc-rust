use std::fmt::{Display, Formatter};
use std::io::Write;
use crate::math::BigInteger;
use crate::{BcError, Result};
use crate::asn1::{Asn1Encodable, Asn1Object, Asn1TaggedObject, Asn1Write, EncodingType};
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::asn1_tags::{INTEGER, UNIVERSAL};
use crate::asn1::primitive_encoding::PrimitiveEncoding;

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct Asn1Integer {
    value: BigInteger
}

impl Asn1Integer {
    pub fn new(value: BigInteger) -> Self {
        Asn1Integer { value }
    }
    pub fn with_buffer(buffer: &[u8]) -> Result<Self> {
        if Self::is_malformed(buffer) {
            return Err(BcError::with_invalid_argument("malformed integer"));
        }
        Ok(Asn1Integer { value: BigInteger::with_buffer(buffer) })
    }
    pub fn with_tagged(tagged_object: &Asn1TaggedObject, declared_explicit: bool) -> Result<Self> {
        todo!();
    }
    pub fn with_buffer_allow_unsafe(buffer: &[u8]) -> Result<Self> {
        if buffer.is_empty() {
            return Err(BcError::with_invalid_argument("buffer len is zero"));
        }
        Ok(Asn1Integer { value: BigInteger::with_buffer(buffer) })
    }
    pub fn with_i64(value: i64) -> Self {
        Asn1Integer {
            value: BigInteger::with_i64(value)
        }
    }
    pub(crate) fn create_primitive(buffer: Vec<u8>) -> Result<Self> {
        Ok(Asn1Integer {
            value: BigInteger::with_buffer(&buffer)
        })
    }
    pub(crate) fn from_asn1_object(asn1_object: &Asn1Object) -> Result<Self> {
        if let Asn1Object::Integer(integer) = asn1_object {
            Ok(integer.clone())
        } else {
            Err(BcError::with_invalid_argument("not an integer object"))
        }
    }
    /// Apply the correct validation for an INTEGER primitive following the BER rules.
    ///
    /// # Arguments
    /// * `bytes` - The raw encoding of the integer.
    ///
    /// # Returns
    /// * `true` if the encoding is malformed, `false` otherwise.
    pub(crate) fn is_malformed(bytes: &[u8]) -> bool {
        match bytes.len() {
            0 => true,
            1 => false,
            _ => (bytes[0] as i8) == (bytes[1] as i8) >> 7,
        }
    }
    pub(crate) fn get_encoding(&self, encoding_type: EncodingType) -> Box<dyn Asn1Encoding> {
        Box::new(PrimitiveEncoding::new(UNIVERSAL, INTEGER, self.get_content(encoding_type)))
    }
    fn get_content(&self, _: EncodingType) -> Vec<u8> {
        self.value.to_vec()
    }
    
}

impl From<BigInteger> for Asn1Integer {
    fn from(value: BigInteger) -> Self {
        Asn1Integer::new(value)
    }
}

impl AsRef<BigInteger> for Asn1Integer {
    fn as_ref(&self) -> &BigInteger {
        &self.value
    }
}

impl Display for Asn1Integer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl Asn1Encodable for Asn1Integer {
    fn encode_to(&self, writer: &mut dyn Write, encoding_type: EncodingType) -> Result<usize> {
        let mut asn1_writer = Asn1Write::new(writer, encoding_type);
        let length = self.get_encoding(encoding_type).encode(&mut asn1_writer)?;
        Ok(length)
    }
}

#[cfg(test)]
mod tests {
    use crate::asn1::{Asn1Encodable, Asn1Integer, Asn1Object};
    use crate::asn1::EncodingType::{Ber, Der};
    use crate::math::BigInteger;
    use crate::util::encoders::hex::to_decode_with_str;

    /// Ensure existing single byte behavior.
    #[test]
    fn test_valid_encoding_single_byte() {
        let raw_i32 = vec![0x10];
        let i = Asn1Integer::with_buffer(&raw_i32).unwrap();
        check_i32_value(&i, 16);
    }
    #[test]
    fn test_valid_encoding_multi_byte() {
        let raw_i32 = vec![0x10, 0xFF];
        let i = Asn1Integer::with_buffer(&raw_i32).expect("error").into();
        check_i32_value(&i, 4351);
    }
    #[test]
    fn test_invalid_encoding_00() {
        let raw_i32 = vec![0x00, 0x10, 0xFF];
        let i = Asn1Integer::with_buffer(&raw_i32);
        assert!(i.is_err());
    }
    #[test]
    fn test_invalid_encoding_ff() {
        let raw_i32 = vec![0xFF, 0x81, 0xFF];
        let i = Asn1Integer::with_buffer(&raw_i32);
        assert!(i.is_err());
    }
    #[test]
    fn test_invalid_encoding_00_32bits() {
        // Check what would pass loose validation fails outside loose validation.
        let raw_i32 = vec![0x00, 0x00, 0x00, 0x00, 0x10, 0xFF];
        let i = Asn1Integer::with_buffer(&raw_i32);
        assert!(i.is_err());
    }
    #[test]
    fn test_invalid_encoding_ff_32bits() {
        // Check what would pass loose validation fails outside loose validation.
        let raw_i32 = vec![0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0xFF];
        let i = Asn1Integer::with_buffer(&raw_i32);
        assert!(i.is_err());
    }
    #[test]
    fn test_loose_valid_encoding_zero_32b_aligned() {
        let raw_i64 = to_decode_with_str("00000010FF000000").unwrap();
        let i = Asn1Integer::with_buffer_allow_unsafe(&raw_i64)
            .expect("error")
            .into();
        check_i64_value(&i, 72997666816);
    }
    #[test]
    fn test_loose_valid_encoding_ff_32b_aligned() {
        let raw_i64 = to_decode_with_str("FFFFFF10FF000000").unwrap();
        let i = Asn1Integer::with_buffer_allow_unsafe(&raw_i64)
            .expect("error")
            .into();
        check_i64_value(&i, -1026513960960);
    }

    #[test]
    fn test_loose_valid_encoding_ff_32b_aligned_1not0() {
        let raw_i64 = to_decode_with_str("FFFEFF10FF000000").unwrap();
        let i = Asn1Integer::with_buffer_allow_unsafe(&raw_i64)
            .expect("error")
            .into();
        check_i64_value(&i, -282501490671616);
    }

    #[test]
    fn test_loose_valid_encoding_ff_32b_aligned_2not0() {
        let raw_i64 = to_decode_with_str("FFFFFE10FF000000").unwrap();
        let i = Asn1Integer::with_buffer_allow_unsafe(&raw_i64)
            .expect("error")
            .into();
        check_i64_value(&i, -2126025588736);
    }
    #[test]
    fn test_over_sized_encoding() {
        // Should pass as loose validation permits 3 leading 0xFF bytes.
        let der_integer = Asn1Integer::with_buffer_allow_unsafe(
            &to_decode_with_str("FFFFFFFE10FF000000000000").unwrap(),
        )
            .unwrap();
        let big_integer = BigInteger::with_buffer(&to_decode_with_str("FFFFFFFE10FF000000000000").unwrap());

        assert_eq!(der_integer.as_ref(), &big_integer);
    }

    #[test]
    fn test_encode() {
        let result_length = 6;
        let result_buffer = vec![0x02, 0x04, 0x07, 0x5B, 0xCD, 0x15];
        let asn1_integer = Asn1Integer::with_i64(123456789);
        {
            let buffer = asn1_integer.get_encoded(Ber).expect("fail");
            assert_eq!(result_length, buffer.len());
            assert_eq!(result_buffer, buffer);
        }
        {
            let buffer = asn1_integer.get_encoded(Der).expect("fail");
            assert_eq!(result_length, buffer.len());
            assert_eq!(result_buffer, buffer);
        }
        {
            let mut buffer = Vec::<u8>::new();
            let length = asn1_integer.encode_to(&mut buffer, Ber).expect("fail");
            assert_eq!(result_length, length);
            assert_eq!(result_buffer, buffer);
        }
        {
            let mut buffer = Vec::<u8>::new();
            let length = asn1_integer.encode_to(&mut buffer, Ber)
                .expect("fail");
            assert_eq!(result_length, length);
            assert_eq!(result_buffer, buffer);
        }
    }
    #[test]
    fn test_parse_asn1_object() {
        let buffer = vec![0x02u8, 0x04, 0x07, 0x5B, 0xCD, 0x15];
        let asn1_object = Asn1Object::from_read(&mut buffer.as_slice()).unwrap();
        assert!(asn1_object.is_integer());
        let integer = asn1_object.as_integer().unwrap();
        check_i64_value(&integer, 123456789);
    }
    fn check_i64_value(obj: &Asn1Integer, n: i64) {
        let val = obj.as_ref();
        assert_eq!(val.as_i64(), n);
    }
    fn check_i32_value(obj: &Asn1Integer, n: i32) {
        let val = obj.as_ref();
        assert_eq!(val.as_i32(), n);
    }
}