use crate::math::BigInteger;

pub trait GlvEndomorphism {
    fn decompose_scalar(&self, k: &BigInteger) -> [BigInteger; 2];
}