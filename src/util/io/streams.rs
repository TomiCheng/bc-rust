use crate::Result;
use std::io::Read;

pub fn read_fully(reader: &mut dyn Read, buffer: &mut [u8]) -> Result<usize> {
    let mut total_read = 0;
    while total_read < buffer.len() {
        let num_read = reader.read(&mut buffer[total_read..])?;
        if num_read == 0 {
            break;
        }
        total_read += num_read;
    }
    Ok(total_read)
}
