use crate::BcError;
use crate::Result;
use crate::util::io::streams::read_fully;
use std::io;
use std::io::Read;

pub(crate) struct DefiniteLengthRead<'a> {
    reader: &'a mut dyn Read,
    limit: usize,
    original_length: usize,
    remaining: usize,
}

impl<'a> DefiniteLengthRead<'a> {
    pub fn new(reader: &'a mut dyn Read, length: usize, limit: usize) -> Self {
        DefiniteLengthRead {
            reader,
            limit,
            original_length: length,
            remaining: length,
        }
    }
    pub(crate) fn read_fully(&mut self) -> Result<Vec<u8>> {
        if self.remaining == 0 {
            return Ok(vec![]);
        }
        // make sure it's safe to do this!
        let limit = self.limit;
        if self.remaining >= limit {
            return Err(BcError::with_io_error(format!(
                "out of bounds length found: {} >= {}",
                self.remaining, limit
            )));
        }

        let mut bytes = vec![0u8; self.remaining];
        let read_length = read_fully(self.reader, &mut bytes)?;
        self.remaining -= read_length;
        if self.remaining != 0 {
            return Err(BcError::with_end_of_stream(format!(
                "DEF length {0} object truncated by {1}",
                self.original_length, self.remaining
            )));
        }
        Ok(bytes)
    }
    pub(crate) fn remaining(&self) -> usize {
        self.remaining
    }
}
impl<'a> Read for DefiniteLengthRead<'a> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Ok(0);
        }
        let len = self.remaining.min(buf.len());

        let length = self.reader.read(&mut buf[..len])?;
        self.remaining -= length;
        Ok(length)
    }
}
