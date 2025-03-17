use std::io;
use std::sync;

use anyhow::Context;

use super::*;

// use super::asn1_tags::{BOOLEAN, FLAGS, GENERALIZED_TIME, INTEGER, NULL, OBJECT_IDENTIFIER, RELATIVE_OID};
// use super::definite_length_read::DefiniteLengthRead;
// use super::DerIntegerImpl;
// use super::{Asn1Object, DerBooleanImpl, DerNullImpl, DerObjectIdentifierImpl};
// use super::Asn1GeneralizedTimeImpl;
use crate::{BcError, Result};

pub struct Asn1Read<'a> {
    reader: &'a mut dyn io::Read,
    limit: usize,
}

impl<'a> Asn1Read<'a> {
    pub fn new(reader: &'a mut dyn io::Read, limit: usize) -> Asn1Read<'a> {
        Asn1Read { reader, limit }
    }

    pub fn read_u8(&mut self) -> Result<u8> {
        let mut buf = [0u8; 1];
        self.reader
            .read(&mut buf)
            .with_context(|| "read u8 fail")?;
        Ok(buf[0])
    }
    pub fn read_object(&mut self) -> Result<sync::Arc<dyn Asn1Object>> {
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

    fn build_object(&mut self, tag_header: u8, tag_no: u32, length: u32) -> Result<sync::Arc<dyn Asn1Object>> {
        let mut def_reader = definite_length_read::DefiniteLengthRead::new(self.reader, length as usize, self.limit);
        if (tag_header as u32 & asn1_tags::FLAGS) == 0 {
            return create_primitive_der_object(tag_no, &mut def_reader);
        }
        todo!();
    }
}

fn read_u8(reader: &mut dyn io::Read) -> Result<u8> {
    let mut buf = [0u8; 1];
    reader
        .read(&mut buf)
        .with_context(|| "read u8 fail")?;
    Ok(buf[0])
}

pub(crate) fn read_tag_number(reader: &mut dyn io::Read, tag_header: u8) -> Result<u32> {
    let mut tag_no = (tag_header & 0x1F) as u32;
    if tag_no == 0x1F {
        let mut b = read_u8(reader)?;

        anyhow::ensure!(b >= 31, BcError::invalid_format("corrupted stream - high tag number < 31 found"));

        tag_no = (b & 0x7F) as u32;

        // X.690-0207 8.1.2.4.2
        // "c) bits 7 to 1 of the first subsequent octet shall not all be zero."
        anyhow::ensure!(tag_no != 0, BcError::invalid_format("corrupted stream - invalid high tag number found"));

        while b & 0x80 != 0 {
            anyhow::ensure!(
                tag_no >> 24 == 0,
                BcError::invalid_format("Tag number more than 31 bits")
            );
            tag_no <<= 7;

            b = read_u8(reader)?;
            tag_no |= (b & 0x7F) as u32;
        }
    }
    Ok(tag_no)
}

pub(crate) fn read_length(
    reader: &mut dyn io::Read,
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
        anyhow::bail!(BcError::invalid_format("invalid long form definite-length 0xFF"));
    } else {
        let octets_count = length & 0x7F;
        let mut octets_pos = 0;

        length = 0;

        loop {
            let octet = read_u8(reader)? as u32;
            anyhow::ensure!(
                (length >> 23) == 0,
                BcError::invalid_format("long form definite-length more than 31 bits")
            );
            length = (length << 8) + octet;
            if {
                octets_pos += 1;
                octets_pos
            } >= octets_count
            {
                break;
            }
        }

        anyhow::ensure!(
            !(length as usize >= limit && !is_parsing),
            BcError::invalid_format(&format!(
                "corrupted stream - out of bounds length found: {} >= {}",
                length, limit
            ))
        );

        Ok(Some(length))
    }
}

pub(crate) fn create_primitive_der_object(
    tag_no: u32,
    reader: &mut definite_length_read::DefiniteLengthRead,
) -> Result<sync::Arc<dyn Asn1Object>> {
    match tag_no {
        asn1_tags::OBJECT_IDENTIFIER => {
            super::der_object_identifier::check_contents_length(reader.get_remaining())?;
            let contents = get_buffer(reader)?;
            return Ok(sync::Arc::new(DerObjectIdentifier::with_primitive(contents)?));
        },
        asn1_tags::RELATIVE_OID => {
            super::asn1_relative_oid::check_contents_length(reader.get_remaining())?;
            let contents = get_buffer(reader)?;
            return Ok(sync::Arc::new(Asn1RelativeOid::with_vec(contents)?));
        }
        _ => {} // Nothing
    }

    let bytes = reader.to_vec()?;
    return match tag_no {
        asn1_tags::BOOLEAN => Ok(sync::Arc::new(DerBoolean::with_primitive(&bytes)?)),
        asn1_tags::INTEGER => Ok(sync::Arc::new(DerInteger::with_primitive(&bytes)?)),
        asn1_tags::NULL => Ok(sync::Arc::new(DerNull::with_primitive(&bytes)?)),
        asn1_tags::GENERALIZED_TIME => Ok(sync::Arc::new(Asn1GeneralizedTime::with_primitive(&bytes)?)),
        _ => anyhow::bail!(format!("unknown tag {tag_no} encountered")),
    };
}

fn get_buffer(def_reader: &mut definite_length_read::DefiniteLengthRead) -> Result<Vec<u8>> {
    return def_reader.to_vec();
}
