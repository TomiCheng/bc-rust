use std::io::Read;
use crate::asn1::{asn1_tags, Asn1BitString, Asn1Boolean, Asn1Integer, Asn1Object};
use crate::{BcError, Result};
use crate::asn1::asn1_tags::FLAGS;
use crate::asn1::definite_length_read::DefiniteLengthRead;

pub struct Asn1Read<'a> {
    reader: &'a mut dyn Read,
    limit: usize,
}

impl<'a> Asn1Read<'a> {
    pub fn new(reader: &'a mut dyn Read, limit: usize) -> Self {
        Asn1Read { reader, limit }
    }

    pub fn read_object(&mut self) -> Result<Asn1Object> {
        let tag_header = self.read_u8()?;
        let tag_no = self.read_tag_number(tag_header)?;
        let length = self.read_length(false)?;
        if let Some(length) = length {
            // definite-length
            return self.build_object(tag_header, tag_no, length);
        }
        todo!()
    }

    fn read_u8(&mut self) -> Result<u8> {
        let mut buffer = [0; 1];
        self.reader.read_exact(&mut buffer)?;
        Ok(buffer[0])
    }
    fn read_tag_number(&mut self, tag_header: u8) -> Result<u32> {
        let mut tag_no = (tag_header & 0x1F) as u32;

        // with tagged object tag number is bottom 5 bits, or stored at the start of the content

        if tag_no == 0x1F {
            let mut b = self.read_u8()?;
            if b < 31 {
                return Err(BcError::with_io_error("corrupted stream - high tag number < 31 found"));
            }

            tag_no = (b & 0x7F) as u32;
            // X.690-0207 8.1.2.4.2
            // "c) bits 7 to 1 of the first subsequent octet shall not all be zero."

            if tag_no == 0 {
                return Err(BcError::with_io_error("corrupted stream - invalid high tag number found"));
            }

            while b & 0x80 != 0 {
                if tag_no >> 24 != 0 {
                    return Err(BcError::with_io_error("corrupted stream - tag number more than 31 bits"));
                }

                tag_no <<= 7;

                b = self.read_u8()?;
                tag_no |= (b & 0x7F) as u32;
            }
        }
        Ok(tag_no)
    }
    fn read_length(&mut self, is_parsing: bool) -> Result<Option<usize>> {
        let mut length = self.read_u8()? as usize;
        if length >> 7 == 0 {
            // definite-length short form
            Ok(Some(length))
        } else if length == 0x80 {
            // indefinite-length
            Ok(None)
        } else if length == 0xFF {
            return Err(BcError::with_invalid_format("invalid long form definite-length 0xFF"));
        } else {
            let octets_count = length & 0x7F;
            let mut octets_pos = 0;

            length = 0;

            loop {
                let octet = self.read_u8()? as usize;
                if length >> 23 != 0 {
                    return Err(BcError::with_invalid_format("long form definite-length more than 31 bits"));
                }

                length = (length << 8) + octet;
                if {
                    octets_pos += 1;
                    octets_pos
                } >= octets_count
                {
                    break;
                }
            }

            if length >= self.limit && !is_parsing {
                return Err(BcError::with_invalid_format(format!("out of bounds length found: {} >= {}", length, self.limit)));
            }
            Ok(Some(length))
        }
    }
    fn build_object(&mut self, tag_header: u8, tag_no: u32, length: usize) -> Result<Asn1Object> {
        // TODO[asn1] Special-case zero length first?
        let mut def_reader = DefiniteLengthRead::new(self.reader, length, self.limit);
        if (tag_header & FLAGS) == 0 {
            return Self::create_primitive_der_object(tag_no as u8, &mut def_reader);
        }
        todo!();
    }
    
    fn create_primitive_der_object(tag_no: u8, reader: &mut DefiniteLengthRead) -> Result<Asn1Object> {
        let bytes = reader.read_fully()?;
        match tag_no {
            asn1_tags::BOOLEAN => Ok(Asn1Object::Boolean(Asn1Boolean::create_primitive(bytes)?)),
            asn1_tags::INTEGER => Ok(Asn1Object::Integer(Asn1Integer::create_primitive(bytes)?)),
            asn1_tags::BIT_STRING => Ok(Asn1Object::BitString(Asn1BitString::create_primitive(bytes)?)),
            // TODO
            _ => Err(BcError::with_invalid_format(format!("Unsupported primitive tag: {}", tag_no))),
        }
    }
}