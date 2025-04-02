use crate::{Error, Result};
use anyhow::ensure;
use crate::asn1::Asn1Object;

// use std::hash::{Hash, Hasher};
// use std::fmt;
// use crate::{invalid_argument, Error, Result};
//
pub struct Asn1BitString {
    //     pad_bits: u8,
    contents: Vec<u8>,
}
//
impl Asn1BitString {
    pub fn create_primitive(contents: Vec<u8>) -> Result<Self> {
        let length = contents.len();
        ensure!(
            length >= 1,
            Error::InvalidInput {
                msg: "truncated BIT STRING detected".to_string(),
                parameter: "contents".to_string(),
            }
        );

        let pad_bits = contents[0];
        if pad_bits > 0 {
            ensure!(
                !(pad_bits > 7 || length < 2),
                Error::InvalidInput {
                    msg: "invalid pad bits detected".to_string(),
                    parameter: "contents".to_string()
                }
            );
            let final_octet = contents[length - 1];
            if final_octet != (final_octet & (0xFF << pad_bits)) {
                //let asn1_object = Asn1BitString::with_contents(contents, false)?;
                //return Ok(sync::Arc::new(asn1_object));
                // TODO: DlBitString
            }
        }
        Ok(Asn1BitString {
            contents
        })
    }
}

//     /// Creates a new `Asn1BitString` with pad bits.
//     /// # Errors
//     /// - Returns an error if `pad_bits` is greater than 7.
//     /// - Returns an error if `data` is empty and `pad_bits` is not 0.
//     /// # Examples
//     /// ```
//     /// use bc_rust::asn1::Asn1BitString;
//     /// let data = vec![0b10101010, 0b11001100];
//     /// let pad_bits = 3;
//     /// let bit_string = Asn1BitString::with_pad_bits(&data, pad_bits).unwrap();
//     /// assert_eq!(bit_string.pad_bits(), pad_bits);
//     /// assert_eq!(bit_string.contents(), &data);
//     /// ```
//     pub fn with_pad_bits(data: &[u8], pad_bits: u8) -> Result<Asn1BitString> {
//         invalid_argument!(
//             pad_bits > 7,
//             "pad_bits must be in the range 0 to 7",
//             "pad_bits"
//         );
//         invalid_argument!(
//             data.len() == 0 && pad_bits != 0,
//             "if 'data' is empty, 'pad_bits' must be 0",
//             "data, pad_bits"
//         );
//
//         Ok(Asn1BitString {
//             contents: data.to_vec(),
//             pad_bits,
//         })
//     }
//     pub fn pad_bits(&self) -> u8 {
//         self.pad_bits
//     }
//     pub fn contents(&self) -> &Vec<u8> {
//         &self.contents
//     }
//     pub fn contents_len(&self) -> usize {
//         self.contents.len()
//     }
//     pub fn is_octet_aligned(&self) -> bool {
//         self.pad_bits() == 0
//     }
// }
//
// impl Default for Asn1BitString {
//     fn default() -> Self {
//         Asn1BitString {
//             contents: vec![],
//             pad_bits: 0,
//         }
//     }
// }
//
// impl PartialEq for Asn1BitString {
//     fn eq(&self, other: &Self) -> bool {
//         if self.contents.len() != other.contents.len() {
//             return false;
//         }
//         if self.contents.len() == 0 {
//             return true;
//         }
//         for i in 0..self.contents.len() {
//             if i == self.contents.len() - 1 {
//                 let left = self.contents[i] & (0xFF << self.pad_bits);
//                 let right = other.contents[i] & (0xFF << other.pad_bits);
//                 if left != right {
//                     return false;
//                 }
//             } else {
//                 if self.contents[i] != other.contents[i] {
//                     return false;
//                 }
//             }
//         }
//         true
//     }
// }
//
// impl Hash for Asn1BitString {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         if self.contents.len() == 0 {
//             return;
//         }
//         let last = self.contents.len() - 1;
//         let last_der = self.contents[last] & (0xFF << self.pad_bits);
//         state.write_u8(self.pad_bits);
//         for i in 0..last {
//             state.write_u8(self.contents[i]);
//         }
//         state.write_u8(last_der);
//     }
// }
//
// // impl fmt::Display for Asn1BitString {
// //     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
// //         let buffer = self
// //             .get_der_encoded()
// //             .expect("DerBitStringImpl::get_der_encoded failed");
// //         write!(f, "#")?;
// //         for byte in buffer {
// //             write!(f, "{:02x}", byte)?;
// //         }
// //         Ok(())
// //     }
// // }
//
// //     pub fn with_named_bits(named_bits: u32) -> Self {
// //         let mut named_bits = named_bits;
// //         if named_bits == 0 {
// //             return Asn1BitString::new(Arc::new(vec![0u8]), EncodingType::Der);
// //         }
// //         let bits = 32 - named_bits.leading_zeros();
// //         let bytes = (bits + 7) / 8;
// //         debug_assert!(0 < bytes && bytes <= 4);
// //         let mut data = vec![0u8; (bytes + 1) as usize];
// //         for i in 1..bytes {
// //             data[i as usize] = named_bits as u8;
// //             named_bits >>= 8;
// //         }
// //
// //         debug_assert!(named_bits & 0xFF != 0);
// //         data[bytes as usize] = named_bits as u8;
// //
// //         let mut pad_bits = 0;
// //         while (named_bits & (1 << pad_bits)) == 0 {
// //             pad_bits += 1;
// //         }
// //
// //         debug_assert!(pad_bits < 8);
// //         data[0] = pad_bits;
// //         Asn1BitString::new(Arc::new(data), EncodingType::Der)
// //     }
// //
// //     pub fn with_contents(contents: &[u8], check: bool) -> Result<Self> {
// //         if check {
// //             anyhow::ensure!(
// //                 contents.len() > 0,
// //                 BcError::invalid_argument("contents must not be empty", "contents")
// //             );
// //             let pad_bits = contents[0];
// //             if pad_bits > 0 {
// //                 anyhow::ensure!(
// //                     contents.len() >= 2,
// //                     BcError::invalid_argument(
// //                         "zero length data with non-zero pad bits",
// //                         "contents"
// //                     )
// //                 );
// //                 anyhow::ensure!(
// //                     pad_bits <= 7,
// //                     BcError::invalid_argument(
// //                         "pad bits cannot be greater than 7 or less than 0",
// //                         "contents"
// //                     )
// //                 );
// //             }
// //         }
// //         Ok(Asn1BitString::new(
// //             Rc::new(contents.to_vec()),
// //             EncodingType::Der,
// //         ))
// //     }
// //     pub(crate) fn contents(&self) -> Arc<Vec<u8>> {
// //         self.contents.clone()
// //     }
// //
// //     pub fn get_bytes(&self) -> Vec<u8> {
// //         if self.contents.len() == 1 {
// //             return vec![];
// //         }
// //         let pad_bits = self.contents[0];
// //         let mut result = self.contents[1..].to_vec();
// //         // DER requires pad bits be zero
// //         let last_index = result.len() - 1;
// //         result[last_index] &= 0xFF << pad_bits;
// //         result
// //     }
// //
// //     // pub(crate) fn with_primitive(contents: &[u8]) -> Result<sync::Arc<dyn Asn1Object>> {
// //     //     let length = contents.len();
// //     //     anyhow::ensure!(
// //     //         length >= 1,
// //     //         BcError::invalid_argument("truncated BIT STRING detected", "contents")
// //     //     );
// //     //     let pad_bits = contents[0];
// //     //     if pad_bits > 0 {
// //     //         anyhow::ensure!(
// //     //             !(pad_bits > 7 || length < 2),
// //     //             BcError::invalid_argument("invalid pad bits detected", "contents")
// //     //         );
// //     //         let final_octet = contents[length - 1];
// //     //         if final_octet != (final_octet & (0xFF << pad_bits)) {
// //     //             let asn1_object = Asn1BitString::with_contents(contents, false)?;
// //     //             return Ok(sync::Arc::new(asn1_object));
// //     //         }
// //     //     }
// //     //     let contents = contents.to_vec();
// //     //     Ok(sync::Arc::new(Asn1BitString::new(
// //     //         Rc::new(contents),
// //     //         EncodingType::Der,
// //     //     )))
// //     // }
// //
// //     pub(crate) fn flatten_bit_string(bit_strings: &[Asn1BitString]) -> Result<Arc<Vec<u8>>> {
// //         let count = bit_strings.len();
// //         if count == 0 {
// //             return Ok(Arc::new(vec![]));
// //         } else if count == 1 {
// //             return Ok(bit_strings[0].contents.clone());
// //         } else {
// //             let last = count - 1;
// //             let mut total_length = 0;
// //             for i in 0..last {
// //                 let element_contents = &bit_strings[i].contents;
// //                 ensure!(
// //                     element_contents[0] == 0,
// //                     BcError::invalid_argument(
// //                         "only the last nested bit string can have padding",
// //                         "bit_string"
// //                     )
// //                 );
// //                 total_length += element_contents.len() - 1;
// //             }
// //
// //             let last_element_contents = &bit_strings[last].contents;
// //             let pad_bits = last_element_contents[0];
// //             total_length += last_element_contents.len();
// //
// //             let mut contents = vec![0u8; total_length];
// //             contents[0] = pad_bits;
// //
// //             let mut pos = 1;
// //             for i in 0..count {
// //                 let element_contents = &bit_strings[i].contents;
// //                 let length = element_contents.len() - 1;
// //                 contents[pos..pos + length].copy_from_slice(&element_contents[1..]);
// //                 pos += length;
// //             }
// //
// //             debug_assert_eq!(pos, total_length);
// //             Ok(Arc::new(contents))
// //         }
// //     }
// // }
// //
//
// // impl Asn1Encodable for Asn1BitString {
// //     fn encode_to_with_encoding(
// //         &self,
// //         writer: &mut dyn io::Write,
// //         encoding_str: &str,
// //     ) -> Result<usize> {
// //         let asn1_encoding = self.get_encoding_with_type(get_encoding_type(encoding_str));
// //         encode_to_with_encoding(writer, encoding_str, asn1_encoding.as_ref())
// //     }
// // }
// //
// // impl Asn1ObjectInternal for Asn1BitString {
// //     fn get_encoding_with_type(&self, encode_type: EncodingType) -> Box<dyn Asn1Encoding> {
// //         match encode_type {
// //             EncodingType::Der => {
// //                 let pad_bits = self.contents[0];
// //                 if pad_bits != 0 {
// //                     let last = self.contents.len() - 1;
// //                     let last_ber = self.contents[last];
// //                     let last_der = last_ber & (0xFF << pad_bits);
// //                     if last_ber != last_der {
// //                         return Box::new(PrimitiveEncodingSuffixed::new(
// //                             UNIVERSAL,
// //                             BIT_STRING,
// //                             self.contents.clone(),
// //                             last_der,
// //                         ));
// //                     }
// //                 }
// //                 Box::new(PrimitiveEncoding::new(
// //                     UNIVERSAL,
// //                     BIT_STRING,
// //                     self.contents.clone(),
// //                 ))
// //             }
// //             EncodingType::Ber => Box::new(PrimitiveEncoding::new(
// //                 UNIVERSAL,
// //                 BIT_STRING,
// //                 self.contents.clone(),
// //             )),
// //             EncodingType::Dl => {
// //                 if let Some(elements) = &self.elements {
// //                     let encodings = get_contents_encodings(encode_type, elements);
// //                     Box::new(ConstructedILEncoding::new(UNIVERSAL, BIT_STRING, encodings))
// //                 } else {
// //                     Box::new(PrimitiveEncoding::new(
// //                         UNIVERSAL,
// //                         BIT_STRING,
// //                         self.contents.clone(),
// //                     ))
// //                 }
// //             }
// //         }
// //     }
// // }
