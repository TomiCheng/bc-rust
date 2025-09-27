
pub(crate) fn inverse_u32(d: u32) -> u32 {
    debug_assert!(d & 1 == 1);

    let mut x = d;
    for _ in 0..4 {
        x = x.wrapping_mul(2u32.wrapping_sub(d.wrapping_mul(x)));
    }

    debug_assert!(d.wrapping_mul(x) == 1);
    x
}

pub(crate) fn inverse_u64(d: u64) -> u64 {
    debug_assert!(d & 1 == 1);

    let mut x = d;
    for _ in 0..5 {
        x = x.wrapping_mul(2u64.wrapping_sub(d.wrapping_mul(x)));
    }

    debug_assert!(d.wrapping_mul(x) == 1);
    x
}