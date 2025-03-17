use std::io;
use std::cell;
use std::ops::SubAssign;

use anyhow::Context;

use crate::util::io::streams;
use crate::{BcError, Result};

pub(crate) struct DefiniteLengthRead<'a> {
    reader: &'a mut dyn io::Read,
    limit: usize,
    length: usize,
    original_length: usize,
    remaining: cell::Cell<usize>,
}

impl DefiniteLengthRead<'_> {
    pub(crate) fn new(reader: &mut dyn io::Read, length: usize, limit: usize) -> DefiniteLengthRead {
        DefiniteLengthRead {
            reader,
            limit,
            length,
            original_length: length,
            remaining: cell::Cell::new(length),
        }
    }

    pub(crate) fn get_remaining(&self) -> usize {
        self.remaining.get()
    }

    pub(crate) fn to_vec(&mut self) -> Result<Vec<u8>> {
        if self.remaining.get() == 0 {
            return Ok(Vec::new());
        }

        anyhow::ensure!(
            self.remaining.get() < self.limit,
            "corrupted stream - out of bounds length found: {0} >= {1}",
            self.remaining.get(),
            self.limit
        );

        let mut bytes = vec![0u8; self.remaining.get()];
        let readed_length =
            streams::read_fully(self.reader, &mut bytes).with_context(|| "error reading bytes")?;

        self.remaining.get_mut().sub_assign(readed_length);

        anyhow::ensure!(
            self.remaining.get() == 0,
            BcError::eof_of_read(&format!(
                "DEF length {0} object truncated by {1}",
                self.original_length,
                self.remaining.get()
            ))
        );

        // todo set_parent_eof_detect();

        return Ok(bytes);
    }
}
