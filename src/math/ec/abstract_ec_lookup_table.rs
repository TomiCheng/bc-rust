use crate::math::ec::EcLookupTable;

pub trait AbstractEcLookupTable: EcLookupTable {
    fn lookup_var(&self, index: usize) -> Option<&Self::Point> {
        self.lookup(index)
    }
}