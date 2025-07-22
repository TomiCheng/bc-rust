#![allow(unused)]
use crate::math::BigInteger;
use crate::util::pack;

type Nat576 = [u64; 9];
type Nat576Ext = [u64; 18];

pub(crate) fn copy_to_u64(x: &Nat576, z: &mut Nat576) {
    for i in 0..x.len() {
        z[i] = x[i];
    }
}
pub(crate) fn create_u64() -> Nat576 {
    [0u64; 9]
}
pub(crate) fn create_u64_ext() -> Nat576Ext {
    [0u64; 18]
}
pub(crate) fn equal_u64(x: &Nat576, y: &Nat576) -> bool {
    for i in (0..x.len()).rev() {
        if x[i] != y[i] {
            return false;
        }
    }
    true
}
pub(crate) fn is_one_u64(x: &Nat576) -> bool {
    if x[0] != 1 {
        return false;
    }
    for i in 1..x.len() {
        if x[i] != 0 {
            return false;
        }
    }
    true
}
pub(crate) fn is_zero_u64(x: &Nat576) -> bool {
    for i in 0..x.len() {
        if x[i] != 0 {
            return false;
        }
    }
    true
}
pub(crate) fn to_big_integer(x: &Nat576) -> BigInteger {
    let mut bs = vec![0u8; 72];
    for i in 0..9 {
        let v = x[i];
        pack::u64_to_be(v, &mut bs[((8 - i) << 3)..])
    }
    BigInteger::with_sign_buffer(1, &bs).unwrap()
}
