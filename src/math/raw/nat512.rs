// todo

pub(crate) fn xor_to_u32(x: &[u32], z: &mut [u32]) {
    for i in (0..16).step_by(4) {
        z[i + 0] ^= x[i + 0];
        z[i + 1] ^= x[i + 1];
        z[i + 2] ^= x[i + 2];
        z[i + 3] ^= x[i + 3];
    }
}
pub(crate) fn xor_to_u64(x: &[u64], z: &mut [u64]) {
    for i in (0..8).step_by(4) {
        z[i + 0] ^= x[i + 0];
        z[i + 1] ^= x[i + 1];
        z[i + 2] ^= x[i + 2];
        z[i + 3] ^= x[i + 3];
    }
}