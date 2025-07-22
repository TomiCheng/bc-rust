pub trait EcLookupTable {
    type Point;
    fn size(&self) -> usize;
    fn lookup(&self, index: usize) -> Option<&Self::Point>;
    fn lookup_var(&self, index: usize) -> Option<&Self::Point>;
}