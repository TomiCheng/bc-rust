use std::io::Read;

use super::asn1_tags::{BOOLEAN, FLAGS, INTEGER, NULL};
use super::{Asn1Object, DerBooleanImpl, DerNullImpl};
use super::DerIntegerImpl;
//use super::{asn1_tags::CONSTRUCTED, DerNull};
use super::definite_length_read::DefiniteLengthRead;
use crate::{BcError, Result};

pub struct Asn1Read<'a> {
    reader: &'a mut dyn Read,
    limit: usize,
}

impl<'a> Asn1Read<'a> {
    pub fn new(reader: &'a mut dyn Read, limit: usize) -> Asn1Read<'a> {
        Asn1Read { reader, limit }
    }

    pub fn read_u8(&mut self) -> Result<u8> {
        let mut buf = [0u8; 1];
        self.reader.read(&mut buf).map_err(|e| BcError::IoError {
            msg: "read u8 fail".to_string(),
            source: e,
        })?;
        Ok(buf[0])
    }
    pub fn read_object(&mut self) -> Result<Asn1Object> {
        let tag_header = self.read_u8()?;
        let tag_no = read_tag_number(self.reader, tag_header)?;
        let length = read_length(self.reader, self.limit, true)?;

        if let Some(length) = length {
            // definite-length
            return self.build_object(tag_header, tag_no, length);
            // todo exception mapping
        }
        // if (tag_header as u32 & CONSTRUCTED) == 0 {
        //     return Err(BcError::IoError {
        //         msg: "indefinite-length primitive encoding encountered".to_string(),
        //         source: std::io::Error::new(
        //             std::io::ErrorKind::InvalidData,
        //             "constructed flag not set",
        //         ),
        //     });
        // }

        todo!();
    }

    fn build_object(&mut self, tag_header: u8, tag_no: u32, length: u32) -> Result<Asn1Object> {
        let mut def_reader = DefiniteLengthRead::new(self.reader, length as usize, self.limit);
        if (tag_header as u32 & FLAGS) == 0 {
            return create_primitive_der_object(tag_no, &mut def_reader);
        }
        todo!();
    }
}

fn read_u8(reader: &mut dyn Read) -> Result<u8> {
    let mut buf = [0u8; 1];
    reader.read(&mut buf).map_err(|e| BcError::IoError {
        msg: "read u8 fail".to_string(),
        source: e,
    })?;
    Ok(buf[0])
}

pub(crate) fn read_tag_number(reader: &mut dyn Read, tag_header: u8) -> Result<u32> {
    let mut tag_no = (tag_header & 0x1F) as u32;
    if tag_no == 0x1F {
        let mut b = read_u8(reader)?;
        if b < 31 {
            return Err(BcError::InvalidFormat(
                "corrupted stream - high tag number < 31 found".to_string(),
            ));
        }

        tag_no = (b & 0x7F) as u32;

        // X.690-0207 8.1.2.4.2
        // "c) bits 7 to 1 of the first subsequent octet shall not all be zero."
        if tag_no == 0 {
            return Err(BcError::InvalidFormat(
                "corrupted stream - invalid high tag number found".to_string(),
            ));
        }

        while b & 0x80 != 0 {
            if (tag_no as u32 >> 24) != 0 {
                return Err(BcError::InvalidFormat(
                    "Tag number more than 31 bits".to_string(),
                ));
            }
            tag_no <<= 7;

            b = read_u8(reader)?;
            tag_no |= (b & 0x7F) as u32;
        }
    }
    Ok(tag_no)
}

pub(crate) fn read_length(
    reader: &mut dyn Read,
    limit: usize,
    is_parsing: bool,
) -> Result<Option<u32>> {
    let mut length = read_u8(reader)? as u32;
    if length >> 7 == 0 {
        // definite-length short form
        return Ok(Some(length));
    } else if length == 0x80 {
        // indefinite-length
        return Ok(None);
    } else if length == 0xFF {
        return Err(BcError::InvalidFormat(
            "invalid long form definite-length 0xFF".to_string(),
        ));
    } else {
        let octets_count = length & 0x7F;
        let mut octets_pos = 0;

        length = 0;

        loop {
            let octet = read_u8(reader)? as u32;
            if (length >> 23) != 0 {
                return Err(BcError::InvalidFormat(
                    "long form definite-length more than 31 bits".to_string(),
                ));
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

        if length as usize >= limit && !is_parsing {
            return Err(BcError::InvalidFormat(format!(
                "corrupted stream - out of bounds length found: {} >= {}",
                length, limit
            )));
        }
        Ok(Some(length))
    }
}

pub(crate) fn create_primitive_der_object(
    tag_no: u32,
    reader: &mut DefiniteLengthRead,
) -> Result<Asn1Object> {
    //     match tag_no {
    //         _ => {} // Nothing
    //     }

    let bytes = reader.to_vec()?;
    return match tag_no {
        BOOLEAN => Ok(DerBooleanImpl::with_primitive(&bytes)?.into()),
        INTEGER => Ok(DerIntegerImpl::with_primitive(&bytes)?.into()),
        NULL => Ok(DerNullImpl::with_primitive(&bytes)?.into()),
        _ => Err(BcError::IoError {
            msg: format!("unknown tag {tag_no} encountered"),
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, ""),
        }),
    };
}

// fn get_buffer(def_reader: &mut DefiniteLengthRead) {
//     let len = def_reader.get_remaining();

// }
