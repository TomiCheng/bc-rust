use crate::math;
use crate::{Error, Result};

pub struct RsaKeyParametersImpl {
    modulus: math::BigInteger,
    exponent: math::BigInteger,
}

impl RsaKeyParametersImpl {
    pub fn new(
        modulus: math::BigInteger,
        exponent: math::BigInteger,
    ) -> Result<Self> {
        if modulus.get_sign_value() <= 0 {
            return Err(Error::with_invalid_input(
                "RSA modulus must be positive".to_owned(),
                "modulus".to_owned(),
            ));
        }
        if exponent.get_sign_value() <= 0 {
            return Err(Error::with_invalid_input(
                "RSA exponent must be positive".to_owned(),
                "exponent".to_owned(),
            ));
        }
        if exponent.i32_value() & 1 == 0 {
            return Err(Error::with_invalid_input(
                "RSA public exponent must be odd".to_owned(),
                "exponent".to_owned(),
            ));
        }
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
