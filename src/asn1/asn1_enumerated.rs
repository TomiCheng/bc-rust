use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
pub struct Asn1Enumerated {
}
impl Hash for Asn1Enumerated {
    fn hash<H: Hasher>(&self, state: &mut H) {
        todo!();
    }
}