use std::rc::Rc;

use super::asn1_encoding::Asn1Encoding;
use super::asn1_object::Asn1ObjectInternal;
use super::asn1_tags::{BIT_STRING, UNIVERSAL};
use super::asn1_write::EncodingType;
use super::primitive_encoding::PrimitiveEncoding;
use super::primitive_encoding_suffixed::PrimitiveEncodingSuffixed;
use super::{Asn1Convertiable, Asn1Encodable, Asn1ObjectImpl};
use crate::{BcError, Result};

pub struct DerBitString {
    contents: Rc<Vec<u8>>,
}

impl DerBitString {
    pub fn new(contents: Rc<Vec<u8>>) -> Self {
        DerBitString {
            contents: Rc::new(contents.to_vec()),
        }
    }

    /// # Errors
    /// - Returns an error if `pad_bits` is greater than 7.
    /// - Returns an error if `contents` is empty and `pad_bits` is not 0.
    pub fn with_pad_bits(contents: &[u8], pad_bits: u8) -> Result<Self> {
        if pad_bits > 7 {
            return Err(BcError::InvalidInput(
                "must be in the range 0 to 7".to_string(),
            ));
        }
        if contents.len() == 0 && pad_bits != 0 {
            return Err(BcError::InvalidInput(
                "if 'contents' is empty, 'pad_bits' must be 0".to_string(),
            ));
        }

        let mut inner_contents = vec![0u8; contents.len() + 1];
        inner_contents[0] = pad_bits;
        inner_contents[1..].copy_from_slice(contents);

        Ok(DerBitString::new(Rc::new(inner_contents)))
    }

    pub fn with_named_bits(named_bits: u32) -> Self {
        let mut named_bits = named_bits;
        if named_bits == 0 {
            return DerBitString::new(Rc::new(vec![0u8]));
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
        DerBitString::new(Rc::new(data))
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
}

impl Asn1Convertiable for DerBitString {
    fn to_asn1_encodable(self: &Rc<Self>) -> Box<dyn Asn1Encodable> {
        Box::new(Asn1ObjectImpl::new(self.clone()))
    }
}

impl Asn1ObjectInternal for DerBitString {
    fn get_encoding_with_type(&self, _encoding: &EncodingType) -> Box<dyn Asn1Encoding> {
        let pad_bits = self.contents[0];
        if pad_bits != 0 {
            let last = self.contents.len() - 1;
            let last_ber = self.contents[last];
            let last_der = last_ber & (0xFF << pad_bits);
            if last_ber != last_der {
                return Box::new(PrimitiveEncodingSuffixed::new(
                    UNIVERSAL,
                    BIT_STRING,
                    self.contents.clone(),
                    last_der,
                ));
            }
        }

        Box::new(PrimitiveEncoding::new(
            UNIVERSAL,
            BIT_STRING,
            self.contents.clone(),
        ))
    }
}
