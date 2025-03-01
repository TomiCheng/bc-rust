use std::io::Write;

pub(crate) struct DerWriteImpl {
    writer: Box<dyn Write>,
    encoding: i32,
}