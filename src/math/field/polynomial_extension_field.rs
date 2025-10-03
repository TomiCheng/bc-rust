use super::*;

pub trait  PolynomialExtensionField: ExtensionField  {
    fn minimal_polynomial(&self) -> &Box<dyn Polynomial>;
}