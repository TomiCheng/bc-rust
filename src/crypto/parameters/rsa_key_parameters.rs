use crate::crypto::parameters;

pub enum RsaKeyParameters {
    Public(parameters::RsaKeyParametersImpl),
    Private(parameters::RsaPrivateCrtKeyParametersImpl),
}

impl RsaKeyParameters {
    pub fn public_key(modulus: crate::math::BigInteger, exponent: crate::math::BigInteger) -> crate::Result<Self> {
        Ok(RsaKeyParameters::Public(parameters::RsaKeyParametersImpl::new(modulus, exponent)?))
    }

    pub fn private_key(
        modulus: crate::math::BigInteger,
        public_exponent: crate::math::BigInteger,
        private_exponent: crate::math::BigInteger,
        p: crate::math::BigInteger,
        q: crate::math::BigInteger,
        d_p: crate::math::BigInteger,
        d_q: crate::math::BigInteger,
        q_inv: crate::math::BigInteger,
    ) -> crate::Result<Self> {
        Ok(RsaKeyParameters::Private(parameters::RsaPrivateCrtKeyParametersImpl::new(
            modulus,
            public_exponent,
            private_exponent,
            p,
            q,
            d_p,
            d_q,
            q_inv,
        )?))
    }
}