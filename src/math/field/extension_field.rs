use super::*;

pub trait ExtensionField: FiniteField {
    fn subfield(&self) -> &Box<dyn FiniteField>;
    fn degree(&self) -> u32;
}
