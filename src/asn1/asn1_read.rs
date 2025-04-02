use super::*;
use crate::asn1::asn1_object_identifier::check_contents_length;
use crate::asn1::asn1_tags::FLAGS;
use crate::asn1::definite_length_read::DefiniteLengthRead;
use crate::Error;
use crate::Result;
use anyhow::{bail, ensure, Context};
use std::io;

pub struct Asn1Read<'a> {
    reader: &'a mut dyn io::Read,
    limit: usize,
}

impl<'a> Asn1Read<'a> {
    pub fn new(reader: &'a mut dyn io::Read, limit: usize) -> Asn1Read<'a> {
        Asn1Read { reader, limit }
    }
    pub fn read_object(&mut self) -> Result<Asn1Object> {
        let tag_header = read_u8(self.reader)?;
        let tag_no = read_tag_number(self.reader, tag_header)?;
        let length = read_length(self.reader, self.limit, false)?;

        if let Some(length) = length {
            // definite-length
            return self.build_object(tag_header, tag_no, length);
        }
        //         // if (tag_header as u32 & CONSTRUCTED) == 0 {
        //         //     return Err(BcError::IoError {
        //         //         msg: "indefinite-length primitive encoding encountered".to_string(),
        //         //         source: std::io::Error::new(
        //         //             std::io::ErrorKind::InvalidData,
        //         //             "constructed flag not set",
        //         //         ),
        //         //     });
        //         // }
        //
        todo!();
    }

    fn build_object(&mut self, tag_header: u8, tag_no: u32, length: u32) -> Result<Asn1Object> {
        // TODO[asn1] Special-case zero length first?

        let mut def_reader = DefiniteLengthRead::new(self.reader, length as usize, self.limit);
        if (tag_header as u32 & FLAGS) == 0 {
            return create_primitive_der_object(tag_no, &mut def_reader);
        }
        //
        todo!();
    }
    // }
    //

    // pub(crate) fn read_length(&mut self, is_parsing: bool) -> Result<Option<u32>> {
    //     let mut length = self.read_u8()? as u32;
    //     if length >> 7 == 0 {
    //         return Ok(Some(length));
    //     }
    //     if length == 0x80 {
    //         return Ok(None);
    //     }
    //     invalid_format!(length == 0xFF, "invalid long form definite-length 0xFF");
    //     let octets_count = length & 0x7F;
    //     let mut octets_pos = 0;
    //
    //     length = 0;
    //
    //     loop {
    //         let octet = self.read_u8()? as u32;
    //         invalid_format!(
    //             (length >> 23) != 0,
    //             "long form definite-length more than 31 bits"
    //         );
    //
    //         length = (length << 8) + octet;
    //         if {
    //             octets_pos += 1;
    //             octets_pos
    //         } >= octets_count
    //         {
    //             break;
    //         }
    //     }
    //
    //     invalid_format!(
    //         !(length as usize >= self.limit && !is_parsing),
    //         "corrupted stream - out of bounds length found: {} >= {}",
    //         length,
    //         self.limit
    //     );
    //
    //     todo!();
    // }
}

pub(crate) fn read_u8(reader: &mut dyn io::Read) -> Result<u8> {
    let mut buf = [0u8; 1];
    reader
        .read_exact(&mut buf)
        .with_context(|| "read u8 fail")?;
    Ok(buf[0])
}

pub(crate) fn read_tag_number(reader: &mut dyn io::Read, tag_header: u8) -> Result<u32> {
    let mut tag_no = (tag_header & 0x1F) as u32;

    // with tagged object tag number is bottom 5 bits, or stored at the start of the content

    if tag_no == 0x1F {
        let mut b = read_u8(reader)?;
        ensure!(
            b >= 31,
            Error::IoError {
                msg: "corrupted stream - high tag number < 31 found".to_string()
            }
        );

        tag_no = (b & 0x7F) as u32;
        // X.690-0207 8.1.2.4.2
        // "c) bits 7 to 1 of the first subsequent octet shall not all be zero."
        ensure!(
            tag_no != 0,
            Error::IoError {
                msg: "corrupted stream - invalid high tag number found".to_string()
            }
        );

        while b & 0x80 != 0 {
            ensure!(
                tag_no >> 24 == 0,
                Error::IoError {
                    msg: "Tag number more than 31 bits".to_string()
                }
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
        Ok(Some(length))
    } else if length == 0x80 {
        // indefinite-length
        Ok(None)
    } else if length == 0xFF {
        bail!(Error::InvalidFormat {
            msg: "invalid long form definite-length 0xFF".to_string()
        });
    } else {
        let octets_count = length & 0x7F;
        let mut octets_pos = 0;

        length = 0;

        loop {
            let octet = read_u8(reader)? as u32;
            ensure!(
                (length >> 23) == 0,
                Error::InvalidFormat {
                    msg: "long form definite-length more than 31 bits".to_string()
                }
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

        ensure!(
            !(length as usize >= limit && !is_parsing),
            Error::InvalidFormat {
                msg: format!(
                    "corrupted stream - out of bounds length found: {} >= {}",
                    length, limit
                )
            }
        );
        Ok(Some(length))
    }
}

pub(crate) fn create_primitive_der_object(
    tag_no: u32,
    reader: &mut DefiniteLengthRead,
) -> Result<Asn1Object> {
    match tag_no {
        asn1_tags::BMP_STRING => {
            return create_der_bmp_string(reader);
        }
        asn1_tags::ENUMERATED => {
            let bytes = reader.read_fully()?;
            return Ok(Asn1Object::Enumerated(Asn1Enumerated::create_primitive(
                bytes,
            )?));
        }
        asn1_tags::OBJECT_IDENTIFIER => {
            check_contents_length(reader.remaining())?;
            let bytes = reader.read_fully()?;
            return Ok(Asn1Object::ObjectIdentifier(
                Asn1ObjectIdentifier::create_primitive(bytes)?,
            ));
        }
        asn1_tags::RELATIVE_OID => {
            asn1_relative_oid::check_contents_length(reader.remaining())?;
            let bytes = reader.read_fully()?;
            return Ok(Asn1Object::RelativeOid(Asn1RelativeOid::create_primitive(
                bytes,
            )?));
        }
        _ => {} // Nothing
    }

    let bytes = reader.read_fully()?;
    match tag_no {
        asn1_tags::BOOLEAN => Ok(Asn1Object::Boolean(Asn1Boolean::create_primitive(bytes)?)),
        asn1_tags::BIT_STRING => Ok(Asn1Object::BitString(Asn1BitString::create_primitive(
            bytes,
        )?)),
        asn1_tags::GENERALIZED_TIME => Ok(Asn1Object::GeneralizedTime(
            Asn1GeneralizedTime::create_primitive(bytes)?,
        )),
        asn1_tags::GENERAL_STRING => Ok(Asn1Object::GeneralString),
        asn1_tags::GRAPHIC_STRING => Ok(Asn1Object::GraphicString),
        asn1_tags::IA5_STRING => Ok(Asn1Object::Ia5String),
        asn1_tags::INTEGER => Ok(Asn1Object::Integer(Asn1Integer::create_primitive(bytes)?)),
        asn1_tags::NULL => Ok(Asn1Object::Null(Asn1Null::create_primitive(bytes)?)),
        asn1_tags::NUMERIC_STRING => Ok(Asn1Object::NumericString),
        asn1_tags::OBJECT_DESCRIPTOR => Ok(Asn1Object::ObjectDescriptor(
            Asn1ObjectDescriptor::create_primitive(bytes)?,
        )),
        asn1_tags::OCTET_STRING => Ok(Asn1Object::OctetString(Asn1OctetString::create_primitive(
            bytes,
        )?)),
        asn1_tags::PRINTABLE_STRING => Ok(Asn1Object::PrintableString),
        asn1_tags::T61_STRING => Ok(Asn1Object::T61String),
        asn1_tags::UNIVERSAL_STRING => Ok(Asn1Object::UniversalString),
        asn1_tags::UTC_TIME => Ok(Asn1Object::UtcTime(Asn1UtcTime::create_primitive(bytes)?)),
        asn1_tags::UTF8_STRING => Ok(Asn1Object::Utf8String),
        asn1_tags::VIDEOTEX_STRING => Ok(Asn1Object::VideotexString),
        asn1_tags::VISIBLE_STRING => Ok(Asn1Object::VisibleString),
        asn1_tags::REAL
        | asn1_tags::EMBEDDED_PDV
        | asn1_tags::TIME
        | asn1_tags::UNRESTRICTED_STRING
        | asn1_tags::DATE
        | asn1_tags::TIME_OF_DAY
        | asn1_tags::DATE_TIME
        | asn1_tags::DURATION
        | asn1_tags::OBJECT_DESCRIPTOR_IRI
        | asn1_tags::RELATIVE_OID_IRI => bail!(Error::IoError {
            msg: format!("unsupported tag : {} encountered", tag_no)
        }),
        _ => bail!(Error::IoError {
            msg: format!("unknown tag : {} encountered", tag_no)
        }),
    }
}

fn create_der_bmp_string(reader: &mut DefiniteLengthRead) -> Result<Asn1Object> {
    todo!()
}
// fn get_buffer<'a>(
//     def_reader: &mut DefiniteLengthRead,
//     temp_buffers: &'a mut [Vec<u8>; 16],
// ) -> Result<(bool, &'a Vec<u8>)> {
//     //return def_reader.to_vec();
//     todo!();
// }
