use crate::Result;
use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::{Asn1Object, EncodingType};
use std::io::Write;

pub struct Asn1Write<'a> {
    writer: &'a mut dyn Write,
    encoding_type: EncodingType,
}

impl<'a> Asn1Write<'a> {
    pub fn new(writer: &'a mut dyn Write, encoding_type: EncodingType) -> Self {
        Asn1Write { writer, encoding_type }
    }

    pub fn write_object(&mut self, asn1_object: &Asn1Object) -> Result<usize> {
        let mut length = 0;
        length += asn1_object.get_encoding(self.encoding_type).encode(self)?;
        self.writer.flush()?;
        Ok(length)
    }

    pub(crate) fn write_identifier(&mut self, tag_class: u8, tag_no: u8) -> Result<usize> {
        let mut tag_no = tag_no;
        let mut length = 0;
        if tag_no < 31 {
            length += self.writer.write(&[tag_class | tag_no])?;
            return Ok(length);
        }

        let mut stack = [0u8; 6];
        let mut pos: isize = stack.len() as isize;

        stack[{
            pos -= 1;
            pos as usize
        }] = tag_no | 0x7f;
        while tag_no > 127 {
            tag_no >>= 7;
            stack[{
                pos -= 1;
                pos as usize
            }] = tag_no & 0x7f | 0x80;
        }
        stack[{
            pos -= 1;
            pos as usize
        }] = tag_class | 0x1F;
        length += self.writer.write(&stack[(pos as usize)..])?;
        Ok(length)
    }
    pub(crate) fn write_dl(&mut self, object_length: usize) -> Result<usize> {
        let mut object_length = object_length;
        let mut length = 0;
        if object_length < 128 {
            length += self.writer.write(&[object_length as u8])?;
            return Ok(length);
        }

        let mut stack = [0u8; 5];
        let mut pos = stack.len() as isize;

        loop {
            stack[{
                pos -= 1;
                pos as usize
            }] = object_length as u8;
            object_length >>= 8;
            if object_length == 0 {
                break;
            }
        }

        let count = stack.len() - pos as usize;
        stack[{
            pos -= 1;
            pos as usize
        }] = (count | 0x80) as u8;
        length += self.writer.write(&stack[(pos as usize)..(pos as usize + count + 1)])?;
        Ok(length)
    }
    pub(crate) fn write(&mut self, bytes: &[u8]) -> Result<usize> {
        let length = self.writer.write(bytes)?;
        Ok(length)
    }
    pub(crate) fn write_u8(&mut self, data: u8) -> Result<usize> {
        let length = self.writer.write(&[data])?;
        Ok(length)
    }
    pub(crate) fn write_encodings(&mut self, encodings: &[Box<dyn Asn1Encoding>]) -> Result<usize> {
        let mut length = 0;
        for encoding in encodings {
            length += encoding.encode(self)?;
        }
        Ok(length)
    }
}

pub fn get_contents_encodings(encoding_type: EncodingType, elements: &[Asn1Object]) -> Vec<Box<dyn Asn1Encoding>> {
    elements.iter().map(|e| e.get_encoding(encoding_type)).collect()
}
pub(crate) fn get_length_of_encoding_dl(tag_no: u8, contents_length: usize) -> usize {
    get_length_of_identifier(tag_no) + get_length_of_dl(contents_length) + contents_length
}
pub(crate) fn get_length_of_identifier(tag_no: u8) -> usize {
    if tag_no < 31 {
        return 1;
    }
    let mut length = 2;
    let mut tag_no = tag_no;
    while { tag_no >>= 7; tag_no } > 0 {
        length += 1;
    }
    length
}
pub(crate) fn get_length_of_dl(contents_length: usize) -> usize {
    if contents_length < 128 {
        return 1;
    }
    let mut length = 2;
    let mut contents_length = contents_length;
    while { contents_length >>= 8; contents_length } > 0 {
        length += 1;
    }
    length
}
pub(crate) fn get_length_of_encodings_il(tag_no: u8, contents_encodings: &[Box<dyn Asn1Encoding>]) -> usize {
    get_length_of_identifier(tag_no) + 3 + get_length_of_contents(contents_encodings)
}
pub(crate) fn get_length_of_contents(contents_encodings: &[Box<dyn Asn1Encoding>]) -> usize {
    contents_encodings.iter().map(|e| e.get_length()).sum()
}
#[cfg(test)]
mod tests {}
