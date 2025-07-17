#[inline]
pub(crate) fn bit_permute_step_u32(x: u32, m: u32, s: u32) -> u32 {
    debug_assert!((m & (m << s)) == 0);
    debug_assert!((m << s) >> s == m);

    let t = (x ^ (x >> s)) & m;
    t ^ (t << s) ^ x
}

#[inline]
pub(crate) fn bit_permute_step_u64(x: u64, m: u64, s: u64) -> u64 {
    debug_assert!((m & (m << s)) == 0);
    debug_assert!((m << s) >> s == m);

    let t = (x ^ (x >> s)) & m;
    t ^ (t << s) ^ x
}

pub(crate) fn bit_permute_step2_u32(hi: &mut u32, lo: &mut u32, m: u32, s: u32) {
    debug_assert!((m & (m << s)) == 0);
    debug_assert!((m << s) >> s == m);

    let t = ((*lo >> s) ^ *hi) & m;
    *lo ^= t << s;
    *hi ^= t;
}

pub(crate) fn bit_permute_step2_u64(hi: &mut u64, lo: &mut u64, m: u64, s: u32) {
    debug_assert!((m & (m << s)) == 0);
    debug_assert!((m << s) >> s == m);

    let t = ((*lo >> s) ^ *hi) & m;
    *lo ^= t << s;
    *hi ^= t;
}

pub(crate) fn bit_permute_step_simple_u32(x: u32, m: u32, s: u32) -> u32 {
    debug_assert!((m << s) == !m);
    debug_assert!((m & !m) == 0);

    ((x & m) << s) | ((x >> s) & m)
}

pub(crate) fn bit_permute_step_simple_u64(x: u64, m: u64, s: u32) -> u64 {
    debug_assert!((m << s) == !m);
    debug_assert!((m & !m) == 0);

    ((x & m) << s) | ((x >> s) & m)
}