pub struct KeyParameter {
    pub(crate) key: Vec<u8>,
}

impl KeyParameter {
    pub fn new() -> Self {
        Self::new_with_length(16)
    }
    pub fn new_with_length(length: usize) -> Self {
        KeyParameter { key: vec![0u8; length] }
    }
    pub fn new_with_buffer(key: &[u8]) -> Self  {
        KeyParameter { key: key.to_vec() }
    }
    pub fn reverse(&self) -> Self {
        let mut key = self.key.clone();
        key.reverse();
        KeyParameter { key }
    }
}
