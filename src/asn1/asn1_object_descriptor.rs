use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
pub struct Asn1ObjectDescriptor {}
impl Hash for Asn1ObjectDescriptor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        todo!();
    }
}
