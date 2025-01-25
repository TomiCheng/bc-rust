use std::hash::{Hash, Hasher};
use std::fmt::{Debug, Formatter, Result};

pub struct DerBoolean {
    value: u8
}

impl DerBoolean {
    pub fn new(value: bool) -> Self {
        DerBoolean {
            value: if value { 0xFF } else { 0x00 }
        }
    }
    pub fn is_true(&self) -> bool {
        self.value != 0x00
    }
    
}

impl Hash for DerBoolean {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl Debug for DerBoolean {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", if self.is_true() { "TRUE" } else { "FALSE" })
    }
}
