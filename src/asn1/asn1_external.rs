use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
pub struct Asn1External {}
impl Hash for Asn1External {
    fn hash<H: Hasher>(&self, state: &mut H) {
        todo!();
    }
}
