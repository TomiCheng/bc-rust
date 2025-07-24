use crate::math::BigInteger;
use crate::math::ec::morphism::ScalarSplitParameters;

pub struct GlvTypeAParameters {
    i: BigInteger,
    lambda: BigInteger,
    split_params: ScalarSplitParameters,
}
impl GlvTypeAParameters {
    pub fn new(i: BigInteger, lambda: BigInteger, split_params: ScalarSplitParameters) -> Self {
        GlvTypeAParameters {
            i,
            lambda,
            split_params,
        }
    }
    pub fn i(&self) -> &BigInteger {
        &self.i
    }
    pub fn lambda(&self) -> &BigInteger {
        &self.lambda
    }
    pub fn split_params(&self) -> &ScalarSplitParameters {
        &self.split_params
    }
}
