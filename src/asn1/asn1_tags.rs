// 0x00: Reserved for use by the encoding rules
pub const BOOLEAN: u32 = 0x01;
pub const INTEGER: u32 = 0x02;
pub const BIT_STRING: u32 = 0x03;
pub const OCTET_STRING: u32 = 0x04;
pub const NULL: u32 = 0x05;
pub const OBJECT_IDENTIFIER: u32 = 0x06;
pub const OBJECT_DESCRIPTOR: u32 = 0x07;
pub const EXTERNAL: u32 = 0x08;
pub const REAL: u32 = 0x09;
pub const ENUMERATED: u32 = 0x0a;
pub const EMBEDDED_PDV: u32 = 0x0b;
pub const UTF8_STRING: u32 = 0x0c;
pub const RELATIVE_OID: u32 = 0x0d;
pub const TIME: u32 = 0x0e;
pub const SEQUENCE: u32 = 0x10;
// 0x0f: Reserved for future editions of this Recommendation | International Standard
pub const SET: u32 = 0x11;
pub const NUMERIC_STRING: u32 = 0x12;
pub const PRINTABLE_STRING: u32 = 0x13;
pub const T61_STRING: u32 = 0x14;
pub const VIDEOTEX_STRING: u32 = 0x15;
pub const IA5_STRING: u32 = 0x16;
pub const UTC_TIME: u32 = 0x17;
pub const GENERALIZED_TIME: u32 = 0x18;
pub const GRAPHIC_STRING: u32 = 0x19;
pub const VISIBLE_STRING: u32 = 0x1a;
pub const GENERAL_STRING: u32 = 0x1b;
pub const UNIVERSAL_STRING: u32 = 0x1c;
pub const UNRESTRICTED_STRING: u32 = 0x1d;
pub const BMP_STRING: u32 = 0x1e;
pub const DATE: u32 = 0x1f;
pub const TIME_OF_DAY: u32 = 0x20;
pub const DATE_TIME: u32 = 0x21;
pub const DURATION: u32 = 0x22;
pub const OBJECT_DESCRIPTOR_IRI: u32 = 0x23;
pub const RELATIVE_OID_IRI: u32 = 0x24;
// 0x25..: Reserved for addenda to this Recommendation | International Standard

pub const CONSTRUCTED: u32 = 0x20;

pub const UNIVERSAL: u32 = 0x00;
pub const APPLICATION: u32 = 0x40;
pub const CONTEXT_SPECIFIC: u32 = 0x80;
pub const PRIVATE: u32 = 0xc0;

pub const FLAGS: u32 = 0xE0;