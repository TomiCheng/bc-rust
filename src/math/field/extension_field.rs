use crate::math::field::FiniteField;

pub trait ExtensionField {
    /// Returns the base field of the extension field.
    fn sub_field(&self) -> &dyn FiniteField;

    /// Returns the degree of the extension.
    fn degree(&self) -> usize;
}