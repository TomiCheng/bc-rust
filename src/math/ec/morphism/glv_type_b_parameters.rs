use crate::math::BigInteger;
use crate::math::ec::morphism::ScalarSplitParameters;

pub struct GlvTypeBParameters {
    beta: BigInteger,
    lambda: BigInteger,
    split_params: ScalarSplitParameters,
}
impl GlvTypeBParameters {
    pub fn new(beta: BigInteger, lambda: BigInteger, split_params: ScalarSplitParameters) -> Self {
        GlvTypeBParameters {
            beta,
            lambda,
            split_params,
        }
    }
    pub fn beta(&self) -> &BigInteger {
        &self.beta
    }
    pub fn lambda(&self) -> &BigInteger {
        &self.lambda
    }
    pub fn split_params(&self) -> &ScalarSplitParameters {
        &self.split_params
    }
}
