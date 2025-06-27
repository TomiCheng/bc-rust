use std::io::Write;
use crate::asn1::{Asn1Object, EncodingType};
use crate::Result;

pub struct Asn1Write<'a> {
    writer: &'a mut dyn Write,
    encoding_type: EncodingType
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
        
        stack[{ pos -= 1; pos as usize }] = tag_no | 0x7f;
        while tag_no > 127 {
            tag_no >>= 7;
            stack[{ pos -= 1; pos as usize }] = tag_no & 0x7f | 0x80;
        }
        stack[{ pos -= 1; pos as usize }] = tag_class | 0x1F;
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
            stack[{ pos -= 1; pos as usize }] = object_length as u8;
            object_length >>= 8;
            if object_length == 0 {
                break;
            }
        };
        
        let count = stack.len() - pos as usize;
        stack[{ pos -= 1; pos as usize }] = (count | 0x80) as u8;
        length += self.writer.write(&stack[(pos as usize)..(pos as usize + count)])?;
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
}

#[cfg(test)]
mod tests {}