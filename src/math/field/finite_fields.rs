use crate::math::BigInteger;
use crate::math::field::generic_polynomial_extension_field::GenericPolynomialExtensionField;
use crate::math::field::gf2_polynomial::Gf2Polynomial;
use crate::math::field::prime_field::PrimeField;
use crate::math::field::{FiniteField, PolynomialExtensionField};
use crate::{BcError, Result};

pub fn create_prime_field(characteristic: BigInteger) -> Result<PrimeField> {
    let bit_length = characteristic.bit_length();
    if characteristic.sign() <= 0 || bit_length < 2 {
        return Err(BcError::with_invalid_argument("must be >= 2"));
    }

    if bit_length < 3 {
        match characteristic.as_i32() {
            2 => return Ok(PrimeField::new(BigInteger::with_i32(2))),
            3 => return Ok(PrimeField::new(BigInteger::with_i32(3))),
            _ => {}
        }
    }
    Ok(PrimeField::new(characteristic))
}
pub fn get_binary_extension_field(exponents: &[i32]) -> Result<impl PolynomialExtensionField> {
    if exponents.is_empty() {
        return Err(BcError::with_invalid_argument(
            "exponents must not be empty",
        ));
    }

    if exponents[0] != 0 {
        return Err(BcError::with_invalid_argument(
            "Irreducible polynomials in GF(2) must have constant term",
        ));
    }

    for i in 1..exponents.len() {
        if exponents[i] <= exponents[i - 1] {
            return Err(BcError::with_invalid_argument(
                "Polynomial exponents must be monotonically increasing",
            ));
        }
    }

    Ok(GenericPolynomialExtensionField::new(
        Box::new(PrimeField::new(BigInteger::with_i32(2))),
        Box::new(Gf2Polynomial::new(exponents.to_vec())),
    ))
}
