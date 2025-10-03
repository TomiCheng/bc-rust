use super::*;
use crate::BcResult;
use crate::err_invalid_arg;
use crate::math::BigInteger;
use crate::math::big_integer::{THREE, TWO};
use std::sync::LazyLock;

static GF_2: LazyLock<PrimeField> = LazyLock::new(|| PrimeField::new(TWO.clone()));
static GF_3: LazyLock<PrimeField> = LazyLock::new(|| PrimeField::new(THREE.clone()));
pub fn get_prime_field(characteristic: BigInteger) -> BcResult<impl FiniteField> {
    let bit_length = characteristic.bit_length();
    err_invalid_arg!(
        characteristic.sign() <= 0 || bit_length < 2,
        "must be >= 2",
        "characteristic"
    );
    Ok(match bit_length {
        2 => GF_2.clone(),
        3 => GF_3.clone(),
        _ => PrimeField::new(characteristic),
    })
}
pub fn get_binary_extension_field(
    exponents: &[u32],
) -> BcResult<Box<dyn PolynomialExtensionField>> {
    err_invalid_arg!(exponents.is_empty(), "must not be empty", "exponents");
    err_invalid_arg!(
        exponents[0] != 0,
        "Irreducible polynomials in GF(2) must have constant term",
        "exponents"
    );
    for i in 1..exponents.len() {
        err_invalid_arg!(
            exponents[i] <= exponents[i - 1],
            "Polynomial exponents must be monotonically increasing",
            "exponents"
        );
    }

    Ok(Box::new(GenericPolynomialExtensionField::new(
        Box::new(GF_2.clone()),
        Box::new(Gf2Polynomial::new(exponents.to_vec())),
    )))
}
