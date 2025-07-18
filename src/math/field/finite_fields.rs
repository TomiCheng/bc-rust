use crate::math::BigInteger;
use crate::math::field::FiniteField;
use crate::{BcError, Result};
use crate::math::field::prime_field::PrimeField;

pub fn get_prime_field(characteristic: BigInteger) -> Result<impl FiniteField> {
    let bit_length = characteristic.bit_length();
    if characteristic.sign() <= 0 || bit_length < 2 {
        return Err(BcError::with_invalid_argument("must be >= 2"));
    }

    // todo cache prime fields
    
    Ok(PrimeField::new(characteristic)) 
}