use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
pub struct Asn1NumericString {}
impl Hash for Asn1NumericString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        todo!();
    }
}