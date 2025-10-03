#![allow(dead_code)]

pub trait BitPermuteStep {
    fn bit_permute_step(x: Self, m: Self, s: Self) -> Self;
    fn bit_permute_step2(hi: &mut Self, lo: &mut Self,  m: Self, s: Self);
    fn bit_permute_step_simple(x: Self, m: Self, s: Self) -> Self;
}

macro_rules! bits_permute_step_macro {
    ($type:ty) => {
        impl BitPermuteStep for $type {
            #[inline]
            fn bit_permute_step(x: Self, m: Self, s: Self) -> Self {
                debug_assert_eq!(m & (m << s), 0);
                debug_assert_eq!((m << s) >> s, m);

                let t = ((x >> s) ^ x) & m;
                x ^ t ^ (t << s)
            }

            #[inline]
            fn bit_permute_step2(hi: &mut Self, lo: &mut Self, m: Self, s: Self) {
                debug_assert_eq!((m << s) >> s, m);

                let t = ((*lo >> s) ^ *hi) & m;
                *hi ^= t;
                *lo ^= t << s;
            }

            #[inline]
            fn bit_permute_step_simple(x: Self, m: Self, s: Self) -> Self {
                debug_assert_eq!(m << s, !m);
                debug_assert_eq!(m & !m, 0);

                ((x & m) << s) | ((x >> s) & m)
            }
        }
    };
}

bits_permute_step_macro!(u32);
bits_permute_step_macro!(u64);

