use crate::BcError;

pub trait Memorable: Clone {
    fn restore(&mut self, other: &Self) -> Result<(), BcError>;
}