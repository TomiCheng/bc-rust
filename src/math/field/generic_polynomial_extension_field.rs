use crate::math::field::{ExtensionField, FiniteField, Polynomial, PolynomialExtensionField};

pub(crate) struct GenericPolynomialExtensionField {
    sub_field: Box<dyn FiniteField>,
    minimal_polynomial: Box<dyn Polynomial>
}

impl GenericPolynomialExtensionField {
    pub fn new(sub_field: Box<dyn FiniteField>, minimal_polynomial: Box<dyn Polynomial>) -> Self {
        GenericPolynomialExtensionField {
            sub_field,
            minimal_polynomial,
        }
    }
    pub fn sub_field(&self) -> &Box<dyn FiniteField> {
        &self.sub_field
    }
    pub fn minimal_polynomial(&self) -> &Box<dyn Polynomial> {
        &self.minimal_polynomial
    }
}

impl ExtensionField for GenericPolynomialExtensionField {
    fn sub_field(&self) -> &dyn FiniteField {
        self.sub_field.as_ref()
    }

    fn degree(&self) -> usize {
        todo!()
    }
}

impl PolynomialExtensionField for GenericPolynomialExtensionField {
}