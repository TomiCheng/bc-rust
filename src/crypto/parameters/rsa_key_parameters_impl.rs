use std::any;

use crate::math;
use crate::{BcError, Result};

pub struct RsaKeyParametersImpl {
    modulus: math::BigInteger,
    exponent: math::BigInteger,
}

impl RsaKeyParametersImpl {
    pub fn new(
        modulus: math::BigInteger,
        exponent: math::BigInteger,
    ) -> Result<Self> {
        anyhow::ensure!(
            modulus.get_sign_value() > 0,
            BcError::invalid_argument("RSA modulus must be positive", "modulus")
        );
        anyhow::ensure!(
            exponent.get_sign_value() > 0,
            BcError::invalid_argument("RSA exponent must be positive", "exponent")
        );
        anyhow::ensure!(
            exponent.i32_value() & 1 != 0 ,
            BcError::invalid_argument("RSA public exponent must be odd", "exponent")
        );
        Ok(RsaKeyParametersImpl {            
            modulus,
            exponent,
        })
    }

    pub fn modulus(&self) -> &math::BigInteger {
        &self.modulus
    }
    pub fn exponent(&self) -> &math::BigInteger {
        &self.exponent
    }
}
