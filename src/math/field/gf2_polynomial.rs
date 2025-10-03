use super::*;

pub(crate) struct Gf2Polynomial {
    exponents: Vec<u32>,
}
impl Gf2Polynomial {
    pub fn new(exponents: Vec<u32>) -> Self {
        Gf2Polynomial { exponents }
    }
}
impl Polynomial for Gf2Polynomial {
    fn degree(&self) -> u32 {
        *self.exponents.last().unwrap()
    }

    fn exponents_present(&self) -> Vec<u32> {
        self.exponents.clone()
    }
}
