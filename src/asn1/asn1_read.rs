use crate::asn1::asn1_tags::{FLAGS, PRIVATE};
use crate::asn1::definite_length_read::DefiniteLengthRead;
use crate::asn1::{
    Asn1BitString, Asn1BmpString, Asn1Boolean, Asn1EncodableVector, Asn1GeneralizedTime, Asn1Ia5String, Asn1Integer, Asn1Null, Asn1Object,
    Asn1ObjectIdentifier, Asn1OctetString, Asn1PrintableString, Asn1RelativeOid, Asn1Sequence, Asn1Set, Asn1TaggedObject, Asn1UtcTime,
    Asn1Utf8String, asn1_tags,
};
use crate::util::io::streams::read_fully;
use crate::{BcError, Result};
use std::io::Read;

pub struct Asn1Read<'a> {
    reader: &'a mut dyn Read,
    limit: usize,
}

impl<'a> Asn1Read<'a> {
    pub fn new(reader: &'a mut dyn Read, limit: usize) -> Self {
        Asn1Read { reader, limit }
    }

    pub fn read_object(&mut self) -> Result<Option<Asn1Object>> {
        let tag_header = self.read_u8();

        if let Err(_) = tag_header {
            return Ok(None);
        }
        let tag_header = tag_header?;
        let tag_no = self.read_tag_number(tag_header)?;
        let length = self.read_length(false)?;
        if let Some(length) = length {
            // definite-length
            let object = self.build_object(tag_header, tag_no, length)?;
            return Ok(Some(object));
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
                return Err(BcError::with_invalid_format(format!(
                    "out of bounds length found: {} >= {}",
                    length, self.limit
                )));
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

        let tag_class = tag_header & PRIVATE;
        if tag_class != 0 {
            let is_constructed = (tag_header & asn1_tags::CONSTRUCTED) != 0;
            return Self::read_tagged_object_dl(tag_class, tag_no, is_constructed, &mut def_reader);
        }
        match tag_no as u8 {
            asn1_tags::SEQUENCE => Ok(Asn1Object::from(Asn1Sequence::from_vector(Self::read_vector_from_definite_length_read(
                &mut def_reader,
            )?)?)),
            asn1_tags::SET => Ok(Asn1Object::from(Asn1Set::from_vector(Self::read_vector_from_definite_length_read(
                &mut def_reader,
            )?)?)),
            _ => Err(BcError::with_io_error(format!("unknown tag 0x{:X} encountered", tag_no))),
        }
    }

    fn create_primitive_der_object(tag_no: u8, reader: &mut DefiniteLengthRead) -> Result<Asn1Object> {
        match tag_no {
            asn1_tags::BMP_STRING => {
                return Ok(create_der_bmp_string(reader)?.into());
            }
            asn1_tags::OBJECT_IDENTIFIER => {
                let bytes = reader.read_fully()?;
                return Ok(Asn1Object::ObjectIdentifier(Asn1ObjectIdentifier::create_primitive(bytes)?));
            }
            _ => {}
        }
        let bytes = reader.read_fully()?;
        match tag_no {
            asn1_tags::BOOLEAN => Ok(Asn1Object::Boolean(Asn1Boolean::create_primitive(bytes)?)),
            asn1_tags::INTEGER => Ok(Asn1Object::Integer(Asn1Integer::create_primitive(bytes)?)),
            asn1_tags::OCTET_STRING => Ok(Asn1Object::OctetString(Asn1OctetString::create_primitive(bytes)?)),
            asn1_tags::BIT_STRING => Ok(Asn1Object::BitString(Asn1BitString::create_primitive(bytes)?)),
            asn1_tags::NULL => Ok(Asn1Object::Null(Asn1Null::create_primitive(bytes)?)),
            asn1_tags::RELATIVE_OID => Ok(Asn1RelativeOid::create_primitive(bytes)?.into()),
            asn1_tags::PRINTABLE_STRING => Ok(Asn1PrintableString::create_primitive(bytes)?.into()),
            asn1_tags::IA5_STRING => Ok(Asn1Object::Ia5String(Asn1Ia5String::create_primitive(bytes)?)),
            asn1_tags::UTC_TIME => Ok(Asn1Object::UtcTime(Asn1UtcTime::create_primitive(bytes)?)),
            asn1_tags::GENERALIZED_TIME => Ok(Asn1GeneralizedTime::create_primitive(bytes)?.into()),
            asn1_tags::UTF8_STRING => Ok(Asn1Utf8String::create_primitive(bytes)?.into()),
            // TODO
            _ => Err(BcError::with_invalid_format(format!("Unsupported primitive tag: 0x{:X}", tag_no))),
        }
    }
    fn read_vector_from_definite_length_read(def_in: &mut DefiniteLengthRead) -> Result<Asn1EncodableVector> {
        let remaining = def_in.remaining();
        if remaining == 0 {
            return Ok(Asn1EncodableVector::with_capacity(0));
        }
        let mut sub_reader = Asn1Read::new(def_in, remaining);
        sub_reader.read_vector()
    }
    fn read_vector(&mut self) -> Result<Asn1EncodableVector> {
        let mut vector = Asn1EncodableVector::new();
        while let Some(o) = self.read_object()? {
            vector.add(o);
        }
        Ok(vector)
    }
    pub(crate) fn read_tagged_object_dl(tag_class: u8, tag_no: u32, is_constructed: bool, def_reader: &mut DefiniteLengthRead) -> Result<Asn1Object> {
        if !is_constructed {
            let contents_octets = def_reader.read_fully()?;
            return Ok(Asn1Object::from(Asn1TaggedObject::create_primitive(
                tag_class,
                tag_no as u8,
                &contents_octets,
            )?));
        }

        let contents_elements = Self::read_vector_from_definite_length_read(def_reader)?;
        Asn1TaggedObject::crate_constructed_dl(tag_class, tag_no as u8, contents_elements)
    }
}

fn create_der_bmp_string(def_in: &mut DefiniteLengthRead) -> Result<Asn1BmpString> {
    let mut remaining_bytes = def_in.remaining();
    if remaining_bytes & 1 != 0 {
        return Err(BcError::with_io_error("malformed BMPString encoding encountered"));
    }
    let length = remaining_bytes / 2;
    let mut chars = vec![0u16; length];
    let mut index = 0;
    let mut buffer = [0u8; 8];
    while remaining_bytes >= 8 {
        if read_fully(def_in, &mut buffer)? != 8 {
            return Err(BcError::with_end_of_stream("EOF encountered in middle of BMPString"));
        }

        chars[index] = u16::from_be_bytes(buffer[0..2].try_into().unwrap());
        index += 1;
        chars[index] = u16::from_be_bytes(buffer[2..4].try_into().unwrap());
        index += 1;
        chars[index] = u16::from_be_bytes(buffer[4..6].try_into().unwrap());
        index += 1;
        chars[index] = u16::from_be_bytes(buffer[6..8].try_into().unwrap());
        index += 1;

        remaining_bytes -= 8;
    }
    if remaining_bytes > 0 {
        if read_fully(def_in, &mut buffer[..remaining_bytes])? != remaining_bytes {
            return Err(BcError::with_end_of_stream("EOF encountered in middle of BMPString"));
        }

        let mut buffer_index = 0;
        loop {
            chars[index] = u16::from_be_bytes(buffer[buffer_index..(buffer_index + 2)].try_into().unwrap());
            index += 1;
            buffer_index += 2;

            if buffer_index >= remaining_bytes {
                break;
            }
        }
    }

    if def_in.remaining() != 0 || chars.len() != index {
        return Err(BcError::with_invalid_operation("BMPString length mismatch"));
    }

    Ok(Asn1BmpString::new(String::from_utf16(&chars)?))
}
