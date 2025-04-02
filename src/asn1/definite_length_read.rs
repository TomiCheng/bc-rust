use crate::util::io::streams::read_fully;
use crate::{Error, Result};
use anyhow::ensure;
use std::io;

pub(crate) struct DefiniteLengthRead<'a> {
    reader: &'a mut dyn io::Read,
    limit: usize,
    original_length: usize,
    remaining: usize,
}

impl<'a> DefiniteLengthRead<'a> {
    pub(crate) fn new(
        reader: &mut dyn io::Read,
        length: usize,
        limit: usize,
    ) -> DefiniteLengthRead {
        DefiniteLengthRead {
            reader,
            limit,
            original_length: length,
            remaining: length,
        }
    }
    pub(crate) fn remaining(&self) -> usize {
        self.remaining
    }
    pub(crate) fn read_fully(&mut self) -> Result<Vec<u8>> {
        if self.remaining == 0 {
            return Ok(Vec::new());
        }

        // make sure it's safe to do this!
        let limit = self.limit;
        ensure!(
            self.remaining < limit,
            Error::IoError {
                msg: format!(
                    "corrupted stream - out of bounds length found: {0} >= {1}",
                    self.remaining, limit
                )
            }
        );

        let mut bytes = vec![0u8; self.remaining];
        let read_length = read_fully(self.reader, &mut bytes)?;
        self.remaining -= read_length;
        ensure!(
            self.remaining == 0,
            Error::EndOfStream {
                msg: format!(
                    "DEF length {0} object truncated by {1}",
                    self.original_length, self.remaining
                )
            }
        );
        Ok(bytes)
    }
}
