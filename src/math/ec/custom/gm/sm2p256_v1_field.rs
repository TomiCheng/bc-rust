use crate::math::raw::nat::{Nat256, Nat256Ext};

const P: Nat256 = Nat256::new([
    0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE,
]);
const P_EXT: Nat256Ext = Nat256Ext::new([
    00000001, 0x00000000, 0xFFFFFFFE, 0x00000001, 0x00000001, 0xFFFFFFFE, 0x00000000, 0x00000002,
    0xFFFFFFFE, 0xFFFFFFFD, 0x00000003, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFE,
]);
const P7: u32 = 0xFFFFFFFE;
const P_EXT_15: u32 = 0xFFFFFFFE;

fn add(x: &Nat256, y: &Nat256, z: &mut Nat256) {
    //let carry = x.add_to(y, z);
    //if carry != 0 || z[7] >= P7 && z >= P {
    //    add_p_inv_to(z)
    //}
}

fn add_p_inv_to(z: &mut Nat256) {
    let mut c = z[0] as i64 + 1;
    z[0] = c as u32;
    c >>= 32;
    if c != 0 {
        c += z[1] as i64;
        z[1] = c as u32;
        c >>= 32;
    }
    c += z[2] as i64 - 1;
    z[2] = c as u32;
    c >>= 32;
    c += z[3] as i64 + 1;
    z[3] = c as u32;
    c >>= 32;
    if c != 0 {
        c += z[4] as i64;
        z[4] = c as u32;
        c >>= 32;
        c += z[5] as i64;
        z[5] = c as u32;
        c >>= 32;
        c += z[6] as i64;
        z[6] = c as u32;
        c >>= 32;
    }

    c += z[7] as i64 + 1;
    z[7] = c as u32;
}