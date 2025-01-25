pub(crate) fn inverse_u32(d: u32) -> u32 {
    debug_assert!(d & 1 == 1);

    //int x = d + (((d + 1) & 4) << 1);   // d.x == 1 mod 2**4
    let mut x = d;  // d.x == 1 mod 2**3
    x = x.wrapping_mul(2u32.wrapping_sub(d.wrapping_mul(x))); // d.x == 1 mod 2**6
    x = x.wrapping_mul(2u32.wrapping_sub(d.wrapping_mul(x))); // d.x == 1 mod 2**12
    x = x.wrapping_mul(2u32.wrapping_sub(d.wrapping_mul(x))); // d.x == 1 mod 2**24
    x = x.wrapping_mul(2u32.wrapping_sub(d.wrapping_mul(x))); // d.x == 1 mod 2**48

    debug_assert!(d.wrapping_mul(x) == 1);
    x
}

pub(crate) fn inverse_u64(d: u64) -> u64 {
    debug_assert!(d & 1 == 1);

    //int x = d + (((d + 1) & 4) << 1);   // d.x == 1 mod 2**4
    let mut x = d;  // d.x == 1 mod 2**3
    x = x.wrapping_mul(2u64.wrapping_sub(d.wrapping_mul(x))); // d.x == 1 mod 2**6
    x = x.wrapping_mul(2u64.wrapping_sub(d.wrapping_mul(x))); // d.x == 1 mod 2**12
    x = x.wrapping_mul(2u64.wrapping_sub(d.wrapping_mul(x))); // d.x == 1 mod 2**24
    x = x.wrapping_mul(2u64.wrapping_sub(d.wrapping_mul(x))); // d.x == 1 mod 2**48
    x = x.wrapping_mul(2u64.wrapping_sub(d.wrapping_mul(x))); // d.x == 1 mod 2**96

    debug_assert!(d.wrapping_mul(x) == 1); 
    x
}