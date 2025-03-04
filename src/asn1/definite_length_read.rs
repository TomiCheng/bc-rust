use std::{any::Any, cell::{Cell, RefCell}, io::Read, ops::{Sub, SubAssign}};

use crate::{util::io::streams::read_fully, BcError, Result};

pub(crate) struct DefiniteLengthRead<'a> {
    reader: &'a mut dyn Read,
    limit: usize,
    length: usize,
    original_length: usize,
    remaining: Cell<usize>,
}

impl DefiniteLengthRead<'_> {
    pub(crate) fn new(reader: &mut dyn Read, length: usize, limit: usize) -> DefiniteLengthRead {
        DefiniteLengthRead {
            reader,
            limit,
            length,
            original_length: length,
            remaining: Cell::new(length),
        }
    }

    pub(crate) fn get_remaining(&self) -> usize {
        self.remaining.get()
    }

    pub(crate) fn to_vec(&mut self) -> Result<Vec<u8>> {
        if self.remaining.get() == 0 {
            return Ok(Vec::new());
        }

        if self.remaining.get() >= self.limit {
            return Err(BcError::IoError {
                msg: format!(
                    "corrupted stream - out of bounds length found: {0} >= {1}",
                    self.remaining.get(), self.limit
                ),
                source: std::io::Error::new(std::io::ErrorKind::InvalidData, "read exceed limit"),
            });
        }

        let mut bytes = vec![0u8;self.remaining.get()];
        let readed_length = read_fully(self.reader, &mut bytes).
            map_err(|e| BcError::IoError {
                msg: "error reading bytes".to_string(),
                source: e,
            })?;
        
        self.remaining.get_mut().sub_assign(readed_length);

        if self.remaining.get() != 0 {
            return Err(BcError::EndOfReadError {
                msg: format!(
                    "DEF length {0} object truncated by {1}",
                    self.original_length,
                    self.remaining.get()
                ),
            });
        }

        // todo set_parent_eof_detect();

        return Ok(bytes);
    }
}
