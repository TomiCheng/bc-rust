use crate::{BcError, Result};

pub struct X509NameTokenizer<'a> {
    value: &'a str,
    separator: char,
    index: usize,
}

impl<'a> X509NameTokenizer<'a> {
    fn new(value: &'a str, separator: char) -> Self {
        X509NameTokenizer { value, separator, index: 0 }
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
        self.index < self.value.len()
    }
    pub fn next_token(&mut self) -> Result<Option<&'a str>> {
        let mut iter = self.value[self.index..].char_indices();
        let mut escaped = false;
        let mut quoted = false;
        let begin_index = self.index;
        let mut length = 0;
        while let Some((i, c)) = iter.next() {
            length = i;
            if escaped {
                escaped = false;
            } else if c == '"' {
                quoted = !quoted;
            } else if quoted {
                // nothing
            } else if c == '\\' {
                escaped = true;
            } else if c == self.separator {
                return Ok(Some(&self.value[begin_index..(begin_index + length)]));
            }
        }

        if escaped || quoted {
            return Err(BcError::with_invalid_format("badly formatted directory string"));
        }
        Ok(Some(&self.value[begin_index..]))
    }
}
