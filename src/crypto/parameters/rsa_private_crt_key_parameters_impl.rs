use crate::math;
use crate::{Error, Result};

#[derive(PartialEq, Hash, Debug)]
pub struct RsaPrivateCrtKeyParametersImpl {
    modulus: math::BigInteger,
    public_exponent: math::BigInteger,
    private_exponent: math::BigInteger,
    p: math::BigInteger,
    q: math::BigInteger,
    d_p: math::BigInteger,
    d_q: math::BigInteger,
    q_inv: math::BigInteger,
}

impl RsaPrivateCrtKeyParametersImpl {
    pub fn new(
        modulus: math::BigInteger,
        public_exponent: math::BigInteger,
        private_exponent: math::BigInteger,
        p: math::BigInteger,
        q: math::BigInteger,
        d_p: math::BigInteger,
        d_q: math::BigInteger,
        q_inv: math::BigInteger,
    ) -> Result<Self> {
        validate_value(&public_exponent, "public_exponent", "exponent")?;
        validate_value(&private_exponent, "private_exponent", "exponent")?;
        validate_value(&p, "p", "P value")?;
        validate_value(&q, "q", "Q value")?;
        validate_value(&d_p, "d_p", "DP value")?;
        validate_value(&d_q, "d_q", "DQ value")?;
        validate_value(&q_inv, "q_inv", "InverseQ value")?;
       
        Ok(RsaPrivateCrtKeyParametersImpl {
            modulus,
            public_exponent,
            private_exponent,
            p,
            q,
            d_p,
            d_q,
            q_inv,
        })
    }

    pub fn modulus(&self) -> &math::BigInteger {
        &self.modulus
    }
    pub fn public_exponent(&self) -> &math::BigInteger {
        &self.public_exponent
    }
    pub fn private_exponent(&self) -> &math::BigInteger {
        &self.private_exponent
    }
    pub fn p(&self) -> &math::BigInteger {
        &self.p
    }
    pub fn q(&self) -> &math::BigInteger {
        &self.q
    }
    pub fn d_p(&self) -> &math::BigInteger {
        &self.d_p
    }
    pub fn d_q(&self) -> &math::BigInteger {
        &self.d_q
    }
    pub fn q_inv(&self) -> &math::BigInteger {
        &self.q_inv
    }
}


fn validate_value(x: &math::BigInteger, name: &str, desc: &str) -> Result<()> {
    if x.get_sign_value() <= 0 {
        return Err(Error::with_invalid_input(
            format!("Not a valid RSA {}", desc),
            name.to_owned(),
        ));
    }
    Ok(())
}