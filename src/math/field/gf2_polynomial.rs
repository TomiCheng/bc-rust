use crate::math::field::Polynomial;

#[derive(Debug, Hash, PartialEq, Eq)]
pub(crate) struct Gf2Polynomial {
    exponents: Vec<i32>,
}

impl Gf2Polynomial {
    pub(crate) fn new(exponents: Vec<i32>) -> Self {
        Gf2Polynomial { exponents }
    }
}

impl Polynomial for Gf2Polynomial {
    fn degree(&self) -> i32 {
        self.exponents[self.exponents.len() - 1]
    }

    fn get_exponents_present(&self) -> Vec<i32> {
        self.exponents.clone()
    }
}