use super::*;
use crate::math::BigInteger;

pub(crate) struct GenericPolynomialExtensionField {
    subfield: Box<dyn FiniteField>,
    polynomial: Box<dyn Polynomial>,
}
impl GenericPolynomialExtensionField {
    pub fn new(subfield: Box<dyn FiniteField>, polynomial: Box<dyn Polynomial>) -> Self {
        Self {
            subfield,
            polynomial,
        }
    }
}
impl FiniteField for GenericPolynomialExtensionField {
    fn characteristic(&self) -> &BigInteger {
        self.subfield.characteristic()
    }

    fn dimension(&self) -> u32 {
        self.subfield.dimension() * self.degree()
    }
}
impl ExtensionField for GenericPolynomialExtensionField {
    fn subfield(&self) -> &Box<dyn FiniteField> {
        &self.subfield
    }
    fn degree(&self) -> u32 {
        self.polynomial.degree()
    }
}
impl PolynomialExtensionField for GenericPolynomialExtensionField {
    fn minimal_polynomial(&self) -> &Box<dyn Polynomial> {
        &self.polynomial
    }
}
