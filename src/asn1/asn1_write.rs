use super::asn1_encodable::{DER, DL};
// use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::util::pack::u32_to_be_bytes;
use crate::Result;
use anyhow::Context;
use std::io;
//use std::sync;
//use std::sync::Arc;

#[derive(Debug, PartialEq, Clone, Copy)]
pub(crate) enum EncodingType {
    Ber,
    Der,
    Dl,
}

pub struct Asn1Write<'a> {
    writer: &'a mut dyn io::Write,
    encoding_type: EncodingType,
}

impl<'a> Asn1Write<'a> {
    fn new(writer: &'a mut dyn io::Write, encoding_type: EncodingType) -> Asn1Write<'a> {
        Asn1Write {
            writer,
            encoding_type,
        }
    }
    pub fn create_with_encoding(writer: &'a mut dyn io::Write, encoding: &str) -> Asn1Write<'a> {
        if encoding == DER {
            Asn1Write::new(writer, EncodingType::Der)
        } else if encoding == DL {
            Asn1Write::new(writer, EncodingType::Dl)
        } else {
            Asn1Write::new(writer, EncodingType::Ber)
        }
    }
    //     pub(crate) fn encoding_type(&self) -> EncodingType {
    //         self.encoding_type
    //     }
    pub(crate) fn write_identifier(&mut self, flags: u32, mut tag_no: u32) -> Result<usize> {
        if tag_no < 31 {
            return Ok(self
                .writer
                .write(&[(flags | tag_no) as u8])
                .with_context(|| "write identifier fail")?);
        }
        let mut stack = [0u8; 6];
        let mut pos = stack.len();
        stack[{
            pos -= 1;
            pos
        }] = (tag_no & 0x7F) as u8;
        while tag_no > 0x7F {
            tag_no >>= 7;
            stack[{
                pos -= 1;
                pos
            }] = (tag_no & 0x7F) as u8;
        }
        stack[{
            pos -= 1;
            pos
        }] = (flags | 0x1F) as u8;
        Ok(self
            .writer
            .write(&stack[pos..])
            .with_context(|| "write identifier fail")?)
    }
    pub(crate) fn write_dl(&mut self, dl: u32) -> Result<usize> {
        if dl < 128 {
            return Ok(self
                .writer
                .write(&[dl as u8])
                .with_context(|| "write dl fail")?);
        }

        let mut encoding = [0u8; 5];
        u32_to_be_bytes(dl, &mut encoding[1..]);
        let leading_zero_bytes = (dl.leading_zeros() / 8) as usize;
        encoding[0] = 0x84 - leading_zero_bytes as u8;
        Ok(self
            .writer
            .write(&encoding[leading_zero_bytes..])
            .with_context(|| "write dl fail")?)
    }
    //     pub(crate) fn encode_contents(
    //         &mut self,
    //         contents_encodings: &[Box<dyn Asn1Encoding>],
    //     ) -> Result<usize> {
    //         let mut length = 0;
    //         for encoding in contents_encodings {
    //             length += encoding.encode(self)?;
    //         }
    //         return Ok(length);
    //     }
    pub fn write(&mut self, data: &[u8]) -> Result<usize> {
        Ok(self.writer.write(data).with_context(|| "write fail")?)
    }
    //     pub fn write_u8(&mut self, data: u8) -> Result<usize> {
    //         return Ok(self
    //             .writer
    //             .write(&[data])
    //             .with_context(|| "write dl fail")?);
}
//
//     //     //     //     pub fn write_encodable(&mut self, asn1_encodable: &dyn Asn1Encodable) -> Result<usize> {
//     //     //     //         let asn1_object = asn1_encodable.to_asn1_object();
//     //     //     //         // let asn1_object_any = &asn1_object as &dyn Any;
//     //     //     //         // if let Some(asn1_object_internal) =
//     //     //     //         //     asn1_object_any.downcast_ref::<&dyn Asn1ObjectInternal>()
//     //     //     //         // {
//     //     //     //         //     let encoding = asn1_object_internal.get_encoding_with_type(self.get_encoding());
//     //     //     //         //     let length = encoding.encode(self)?;
//     //     //     //         //     self.flush_internal();
//     //     //     //         //     return Ok(length);
//     //     //     //         // }
//     //     //     //         return Err(BcError::InvalidCase("write_encodable fail".to_string()));
//     //     //     //     }
//     //     //     //     pub fn write_object(&mut self, asn1_object: &Box<dyn Asn1Object>) -> Result<usize> {
//     //     //     //         let asn1_object_any = asn1_object as &dyn Any;
//     //     //     //         if let Some(asn1_object_internal) =
//     //     //     //             asn1_object_any.downcast_ref::<&dyn Asn1ObjectInternal>()
//     //     //     //         {
//     //     //     //             let encoding = asn1_object_internal.get_encoding_with_type(self.get_encoding());
//     //     //     //             let length = encoding.encode(self)?;
//     //     //     //             self.flush_internal();
//     //     //     //             return Ok(length);
//     //     //     //         }
//     //     //     //         return Err(BcError::InvalidCase("write_encodable fail".to_string()));
//     //     //     //     }
//     //     //     //     fn flush_internal(&mut self) {}
// }

pub(crate) fn get_length_of_encoding_dl(tag_no: u32, contents_length: usize) -> usize {
    get_length_of_identifier(tag_no) + get_length_of_dl(contents_length) + contents_length
}
pub(crate) fn get_length_of_identifier(mut tag_no: u32) -> usize {
    if tag_no < 31 {
        return 1;
    }
    let mut length = 2;
    loop {
        tag_no >>= 7;
        if tag_no > 0 {
            length += 1;
        }
        if tag_no == 0 {
            break;
        }
    }
    return length;
}
pub(crate) fn get_length_of_dl(mut dl: usize) -> usize {
    if dl < 128 {
        return 1;
    }
    let mut length = 2;
    loop {
        dl >>= 8;
        if dl > 0 {
            length += 1;
        }
        if dl == 0 {
            break;
        }
    }
    return length;
}
// pub(crate) fn get_length_of_encoding_il(
//     tag_no: u32,
//     contents_encoding: &dyn Asn1Encoding,
// ) -> usize {
//     get_length_of_identifier(tag_no) + 3 + contents_encoding.get_length()
// }
// pub(crate) fn get_length_of_encodings_il(
//     tag_no: u32,
//     contents_encodings: &[Box<dyn Asn1Encoding>],
// ) -> usize {
//     get_length_of_identifier(tag_no) + 3 + get_length_of_contents(contents_encodings)
// }
// pub(crate) fn get_length_of_contents(contents_encodings: &[Box<dyn Asn1Encoding>]) -> usize {
//     let mut length = 0;
//     for content in contents_encodings {
//         length += content.get_length();
//     }
//     return length;
// }
pub(crate) fn get_encoding_type(encoding: &str) -> EncodingType {
    if encoding == DER {
        EncodingType::Der
    } else if encoding == DL {
        EncodingType::Dl
    } else {
        EncodingType::Ber
    }
}
//
// pub(crate) fn get_contents_encodings(
//     encode_type: EncodingType,
// ) -> Vec<Box<dyn Asn1Encoding>> {
//     let mut encodings = Vec::new();
//     for element in elements {
//         encodings.push(element.get_encoding_with_type(encode_type));
//     }
//     encodings
// }
