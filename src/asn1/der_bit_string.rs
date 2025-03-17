use std::fmt;
use std::io;
use std::sync;

// use super::asn1_encoding::Asn1Encoding;
// use super::asn1_object::{encode_to_with_encoding, get_encoded_with_encoding};
// use super::asn1_tags::{BIT_STRING, UNIVERSAL};
// use super::asn1_write::{get_encoding_type, EncodingType};
//use super::primitive_encoding;
// use super::primitive_encoding_suffixed;
// use super::{Asn1Encodable, Asn1Object};
use super::*;
use crate::{BcError, Result};

#[derive(Clone, Debug)]
pub struct DerBitString {
    contents: sync::Arc<Vec<u8>>,
}

impl DerBitString {
    pub fn new(contents: sync::Arc<Vec<u8>>) -> Self {
        DerBitString { contents }
    }

    /// # Errors
    /// - Returns an error if `pad_bits` is greater than 7.
    /// - Returns an error if `contents` is empty and `pad_bits` is not 0.
    pub fn with_pad_bits(contents: &[u8], pad_bits: u8) -> Result<Self> {
        anyhow::ensure!(
            pad_bits <= 7,
            BcError::invalid_argument("must be in the range 0 to 7", "pad_bits")
        );
        anyhow::ensure!(
            !(contents.len() == 0 && pad_bits != 0),
            BcError::invalid_argument(
                "if 'contents' is empty, 'pad_bits' must be 0",
                "contents, pad_bits"
            )
        );

        let mut inner_contents = vec![0u8; contents.len() + 1];
        inner_contents[0] = pad_bits;
        inner_contents[1..].copy_from_slice(contents);

        Ok(DerBitString::new(std::sync::Arc::new(inner_contents)))
    }

    pub fn with_named_bits(named_bits: u32) -> Self {
        let mut named_bits = named_bits;
        if named_bits == 0 {
            return DerBitString::new(std::sync::Arc::new(vec![0u8]));
        }
        let bits = 32 - named_bits.leading_zeros();
        let bytes = (bits + 7) / 8;
        debug_assert!(0 < bytes && bytes <= 4);
        let mut data = vec![0u8; (bytes + 1) as usize];
        for i in 1..bytes {
            data[i as usize] = named_bits as u8;
            named_bits >>= 8;
        }

        debug_assert!(named_bits & 0xFF == 0);
        data[bytes as usize] = named_bits as u8;

        let mut pad_bits = 0;
        while (named_bits & (1 << pad_bits)) == 0 {
            pad_bits += 1;
        }

        debug_assert!(pad_bits < 8);
        data[0] = pad_bits;
        DerBitString::new(std::sync::Arc::new(data))
    }

    pub fn get_bytes(&self) -> Vec<u8> {
        if self.contents.len() == 1 {
            return vec![];
        }
        let pad_bits = self.contents[0];
        let mut result = self.contents[1..].to_vec();
        // DER requires pad bits be zero
        let last_index = result.len() - 1;
        result[last_index] &= 0xFF << pad_bits;
        result
    }
    fn get_encoding_with_type(
        &self,
        _encode_type: asn1_write::EncodingType,
    ) -> Box<dyn asn1_encoding::Asn1Encoding> {
        let pad_bits = self.contents[0];
        if pad_bits != 0 {
            let last = self.contents.len() - 1;
            let last_ber = self.contents[last];
            let last_der = last_ber & (0xFF << pad_bits);
            if last_ber != last_der {
                return Box::new(primitive_encoding_suffixed::PrimitiveEncodingSuffixed::new(
                    asn1_tags::UNIVERSAL,
                    asn1_tags::BIT_STRING,
                    self.contents.clone(),
                    last_der,
                ));
            }
        }

        return Box::new(primitive_encoding::PrimitiveEncoding::new(
            asn1_tags::UNIVERSAL,
            asn1_tags::BIT_STRING,
            self.contents.clone(),
        ));
    }
}

impl fmt::Display for DerBitString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let buffer = self
            .get_der_encoded()
            .expect("DerBitStringImpl::get_der_encoded failed");
        write!(f, "#")?;
        for byte in buffer {
            write!(f, "{:02x}", byte)?;
        }
        Ok(())
    }
}
impl Asn1Encodable for DerBitString {
    fn get_encoded_with_encoding(&self, encoding_str: &str) -> Result<Vec<u8>> {
        let encoding = self.get_encoding_with_type(asn1_write::get_encoding_type(encoding_str));
        asn1_object::get_encoded_with_encoding(encoding_str, encoding.as_ref())
    }

    fn encode_to_with_encoding(
        &self,
        writer: &mut dyn io::Write,
        encoding_str: &str,
    ) -> Result<usize> {
        let asn1_encoding =
            self.get_encoding_with_type(asn1_write::get_encoding_type(encoding_str));
        asn1_object::encode_to_with_encoding(writer, encoding_str, asn1_encoding.as_ref())
    }
}
