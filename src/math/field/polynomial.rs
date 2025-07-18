pub trait Polynomial {
    fn degree(&self) -> i32;
    fn get_exponents_present(&self) -> Vec<i32>;
}