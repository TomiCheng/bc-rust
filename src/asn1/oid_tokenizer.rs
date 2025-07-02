use std::str::Split;

pub struct OidTokenizer<'a> {
    iter: Split<'a, char>,
}

impl<'a> OidTokenizer<'a> {
    pub fn new(oid: &'_ str) -> OidTokenizer<'_> {
        OidTokenizer {  iter: oid.split('.') }
    }
}

impl<'a> Iterator for OidTokenizer<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}