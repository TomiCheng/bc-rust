use std::io::Read;

pub(crate) struct LimitedRead<'a> {
    reader: &'a mut dyn Read,
    limit: usize,
}

impl<'a> LimitedRead<'a> {
    pub(crate) fn new (reader: &'a mut dyn Read, limit: usize) -> Self {
        LimitedRead { reader, limit }
    }

    pub(crate) fn get_limit(&self) -> usize {
        self.limit
    }

}