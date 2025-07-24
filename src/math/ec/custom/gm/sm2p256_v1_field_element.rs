use std::ops::Add;
use crate::{BcError, Result};
use crate::math::BigInteger;
use crate::math::raw::nat::Nat256;
use crate::util::encoders::hex;
use std::sync::LazyLock;

pub static Q: LazyLock<BigInteger> = LazyLock::new(|| {
    BigInteger::with_sign_buffer(
        1,
        hex::to_decode_with_str("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF")
            .unwrap()
            .as_ref(),
    )
    .unwrap()
});
pub(crate) const P: Nat256 = Nat256::new([0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE]);
const P7: u32 = 0xFFFFFFFE;

pub(crate) struct Sm2p256V1FieldElement {
    x: Nat256,
}
impl Sm2p256V1FieldElement {
    fn new(x: Nat256) -> Self {
        Sm2p256V1FieldElement { x }
    }
    pub fn create() -> Self {
        Self::new(Nat256::create())
    }
    pub fn with_big_integer(x: &BigInteger) -> Result<Self> {
        if x.sign() < 0 || x >= &*Q {
            return Err(BcError::with_invalid_argument("value invalid for SM2P256V1FieldElement"))
        }
        let mut z = Nat256::from_big_integer(&x)?;
        if z[7] >= P7 && z >= P {
            z -= P;
        }
        Ok(Self::new(z))
    }

    // override
    pub fn is_zero(&self) -> bool {
        self.x.is_zero()
    }
    pub fn is_one(&self) -> bool {
        self.x.is_one()
    }
    pub fn test_bit_zero(&self) -> bool {
        self.x.get_bit(0) == 1
    }
    pub fn to_big_integer(&self) -> BigInteger {
        self.x.to_big_integer()
    }
    pub fn field_name(&self) -> String {
        "SM2P256V1Field".to_string()
    }
    pub fn field_size(&self) -> usize {
        (*Q).bit_length()
    }
    pub fn add(&self, b: &Self) -> Self {
        let z = &self.x + &b.x;
        Self::new(z)
    }
}
