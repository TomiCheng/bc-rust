use crate::{BcError, Result};

pub struct X509NameTokenizer<'a> {
    value: &'a str,
    separator: char,
    index: isize,
    count: usize,
}

impl<'a> X509NameTokenizer<'a> {
    fn new(value: &'a str, separator: char) -> Self {
        X509NameTokenizer {
            value,
            separator,
            index: if value.chars().count() < 1 { 0 } else { -1 },
            count: value.chars().count(),
        }
    }
    pub fn with_str(value: &'a str) -> Self {
        X509NameTokenizer::new(value, ',')
    }
    pub fn with_str_and_separator(value: &'a str, separator: char) -> Result<Self> {
        if separator == '"' || separator == '\\' {
            return Err(BcError::with_invalid_format("reserved separator character"));
        }

        Ok(X509NameTokenizer::new(value, separator))
    }
    pub fn has_more_tokens(&self) -> bool {
        self.index < self.count as isize
    }
    pub fn next_token(&mut self) -> Result<Option<String>> {
        if self.index >= self.count as isize {
            return Ok(None);
        }

        let mut quoted = false;
        let mut escaped = false;
        let begin_index = (self.index + 1) as usize;

        while {
            self.index += 1;
            self.index
        } < self.count as isize
        {
            let c = self.value.chars().nth(self.index as usize).unwrap();
            if escaped {
                escaped = false;
            } else if c == '"' {
                quoted = !quoted;
            } else if quoted {
                // nothing
            } else if c == '\\' {
                escaped = true;
            } else if c == self.separator {
                let length = self.index as usize - begin_index;
                return Ok(Some(self.value.chars().skip(begin_index).take(length).collect()));
            }
        }
        if escaped || quoted {
            return Err(BcError::with_invalid_format("badly formatted directory string"));
        }
        let length = self.index as usize - begin_index;
        Ok(Some(self.value.chars().skip(begin_index).take(length).collect()))
    }

    // pub fn next_token(&mut self) -> Result<Option<&'a str>> {
    //     let mut iter = self.value[self.index..].char_indices();
    //     let mut escaped = false;
    //     let mut quoted = false;
    //     let begin_index = self.index;
    //     let mut length = 0;
    //     while let Some((i, c)) = iter.next() {
    //         length = i + 1;
    //         if escaped {
    //             escaped = false;
    //         } else if c == '"' {
    //             quoted = !quoted;
    //         } else if quoted {
    //             // nothing
    //         } else if c == '\\' {
    //             escaped = true;
    //         } else if c == self.separator {
    //             self.index = begin_index + length;
    //             return Ok(Some(&self.value[begin_index..(begin_index + length - 1)]));
    //         }
    //     }
    //
    //     if escaped || quoted {
    //         return Err(BcError::with_invalid_format("badly formatted directory string"));
    //     }
    //
    //     self.index += length;
    //     if length == 0 {
    //         Ok(None)
    //     } else {
    //         Ok(Some(&self.value[begin_index..]))
    //     }
    // }
}
