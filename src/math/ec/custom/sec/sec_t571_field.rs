use crate::math::raw::nat::Nat576;

const M59: u64 = u64::MAX >> 5;
const ROOT_Z: Nat576 = Nat576::new([
    0x2BE1195F08CAFB99,
    0x95F08CAF84657C23,
    0xCAF84657C232BE11,
    0x657C232BE1195F08,
    0xF84657C2308CAF84,
    0x7C232BE1195F08CA,
    0xBE1195F08CAF8465,
    0x5F08CAF84657C232,
    0x784657C232BE119,
]);

pub(crate) fn add(x: &Nat576, y: &Nat576, z: &mut Nat576) {
    x.xor(y, z);
}