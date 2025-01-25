use std::io::{Error, ErrorKind, Result};

pub struct DerInteger {
    buffer: Vec<u8>,
}

impl DerInteger {
    const SIGN_EXT_SIGNED: i32 = -1;
    

    pub fn new_with_i32(value: i32) -> DerInteger {
        DerInteger {
            buffer: value.to_ne_bytes().to_vec(),
        }
    }
    pub fn new_with_i64(value: i64) -> DerInteger {
        DerInteger {
            buffer : value.to_ne_bytes().to_vec(),
        }
    }
    pub fn new_with_buffer(buffer: &[u8]) -> DerInteger {
        DerInteger {
            buffer: buffer.to_vec(),
        }
    }
    pub fn get_i32_value_exact(&self) -> Result<i32> {
        if self.buffer.len() > size_of::<i32>() {
            return Err(Error::new(ErrorKind::InvalidData, "ASN.1 Integer out of i32 range"));
        }
        Ok(Self::i32_value(&self.buffer, Self::SIGN_EXT_SIGNED))
    }
    pub fn get_i64_value_exact(&self) -> Result<i64> {
        if self.buffer.len() > size_of::<i32>() {
            return Err(Error::new(ErrorKind::InvalidData, "ASN.1 Integer out of i64 range"));
        }
        Ok(Self::i64_value(&self.buffer, Self::SIGN_EXT_SIGNED))
    }
    pub(crate) fn i32_value(bytes: &[u8], sign_ext: i32) -> i32 {
        let len = bytes.len();
        let mut pos = std::cmp::max(0, len - size_of::<i32>());
        let mut val = ((bytes[pos] as i8) & sign_ext as i8) as i32;
        pos += 1;
        while pos < len {
            val = (val << 8) | (bytes[pos] as i32);
            pos += 1;
        }
        val
    }
    pub(crate) fn i64_value(bytes: &[u8], sign_ext: i32) -> i64 {
        let len = bytes.len();
        let mut pos = std::cmp::max(0, len - size_of::<i64>());
        let mut val = ((bytes[pos] as i8) & sign_ext as i8) as i64;
        pos += 1;
        while pos < len {
            val = (val << 8) | (bytes[pos] as i64);
            pos += 1;
        }
        val
    }
}