pub trait Polynomial {
    fn degree(&self) -> u32;
    fn exponents_present(&self) -> Vec<u32>;
}