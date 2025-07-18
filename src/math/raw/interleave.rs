use crate::math::raw::bits::{bit_permute_step_u32, bit_permute_step_u64};

const M32: u64 = 0x55555555;
const M64: u64 = 0x5555555555555555;
const M64R: u64 = 0xAAAAAAAAAAAAAAAA;

pub(crate) fn expend_u8_to_u16(x: u8) -> u16 {
    let mut t = x as u16;
    t = (t | (t << 4)) & 0x0F0F;
    t = (t | (t << 2)) & 0x3333;
    t = (t | (t << 1)) & 0x5555;
    t
}

pub(crate) fn expend_u16_to_u32(x: u16) -> u32 {
    let mut t = x as u32;
    t = (t | (t << 8)) & 0x00FF00FF;
    t = (t | (t << 4)) & 0x0F0F0F0F;
    t = (t | (t << 2)) & 0x33333333;
    t = (t | (t << 1)) & 0x55555555;
    t
}

pub(crate) fn expend_u32_to_u64(x: u32) -> u64 {
    let mut t = x;
    t = bit_permute_step_u32(t, 0x0000FF00, 8);
    t = bit_permute_step_u32(t, 0x00F000F0, 4);
    t = bit_permute_step_u32(t, 0x0C0C0C0C, 2);
    t = bit_permute_step_u32(t, 0x22222222, 1);
    ((t as u64 >> 1) & M32) << 32 | (t as u64 & M32)
}

pub(crate) fn expend_u64_to_u128(x: u64) -> u128 {
    let mut t = x;
    t = bit_permute_step_u64(t, 0x00000000FFFF0000, 16);
    t = bit_permute_step_u64(t, 0x0000FF000000FF00, 8);
    t = bit_permute_step_u64(t, 0x00F000F000F000F0, 4);
    t = bit_permute_step_u64(t, 0x0C0C0C0C0C0C0C0C, 2);
    t = bit_permute_step_u64(t, 0x2222222222222222, 1);
    (t & M64) as u128 | ((t >> 1) & M64) as u128
}

pub(crate) fn expend_u64_to_u128_reverse(x: u64) -> u128 {
    let mut t = x;
    t = bit_permute_step_u64(t, 0x00000000FFFF0000, 16);
    t = bit_permute_step_u64(t, 0x0000FF000000FF00, 8);
    t = bit_permute_step_u64(t, 0x00F000F000F000F0, 4);
    t = bit_permute_step_u64(t, 0x0C0C0C0C0C0C0C0C, 2);
    t = bit_permute_step_u64(t, 0x2222222222222222, 1);
    (t & M64R) as u128 | ((t >> 1) & M64R) as u128
}

pub(crate) fn shuffle_u32(x: u32) -> u32 {
    let mut t = x;
    t = bit_permute_step_u32(t, 0x0000FF00, 8);
    t = bit_permute_step_u32(t, 0x00F000F0, 4);
    t = bit_permute_step_u32(t, 0x0C0C0C0C, 2);
    t = bit_permute_step_u32(t, 0x22222222, 1);
    t
}

pub(crate) fn shuffle_u64(x: u64) -> u64 {
    let mut t = x;
    t = bit_permute_step_u64(t, 0x00000000FFFF0000, 16);
    t = bit_permute_step_u64(t, 0x0000FF000000FF00, 8);
    t = bit_permute_step_u64(t, 0x00F000F000F000F0, 4);
    t = bit_permute_step_u64(t, 0x0C0C0C0C0C0C0C0C, 2);
    t = bit_permute_step_u64(t, 0x2222222222222222, 1);
    t
}

pub(crate) fn shuffle2_u32(x: u32) -> u32 {
    let mut t = x;
    t = bit_permute_step_u32(t, 0x0000F0F0, 12);
    t = bit_permute_step_u32(t, 0x00CC00CC, 6);
    t = bit_permute_step_u32(t, 0x22222222, 1);
    t = bit_permute_step_u32(t, 0x0C0C0C0C, 2);
    t
}

pub(crate) fn shuffle2_u64(x: u64) -> u64 {
    let mut t = x;
    t = bit_permute_step_u64(t, 0x00000000FF00FF00, 24);
    t = bit_permute_step_u64(t, 0x0000F0F00000F0F0, 12);
    t = bit_permute_step_u64(t, 0x00CC00CC00CC00CC, 6);
    t = bit_permute_step_u64(t, 0x0A0A0A0A0A0A0A0A, 3);
    t
}

pub(crate) fn unshuffle_u32(x: u32) -> u32 {
    let mut t = x;
    t = bit_permute_step_u32(t, 0x22222222, 1);
    t = bit_permute_step_u32(t, 0x0C0C0C0C, 2);
    t = bit_permute_step_u32(t, 0x00F000F0, 4);
    t = bit_permute_step_u32(t, 0x0000FF00, 8);
    t
}

pub(crate) fn unshuffle_u64(x: u64) -> u64 {
    let mut t = x;
    t = bit_permute_step_u64(t, 0x2222222222222222, 1);
    t = bit_permute_step_u64(t, 0x0C0C0C0C0C0C0C0C, 2);
    t = bit_permute_step_u64(t, 0x00F000F000F000F0, 4);
    t = bit_permute_step_u64(t, 0x0000FF000000FF00, 8);
    t = bit_permute_step_u64(t, 0x00000000FFFF0000, 16);
    t
}

pub(crate) fn unshuffle_u64_to_odd_even(x: u64) -> (u32, u32) {
    let u0 = unshuffle_u64(x);
    let even = (u0 & 0x00000000FFFFFFFF) as u32;
    let odd = (u0 >> 32) as u32;
    (odd, even)
}

pub(crate) fn unshuffle_2u64_to_odd_even(x0: u64, x1: u64) -> (u64, u64) {
    let u0 = unshuffle_u64(x0);
    let u1 = unshuffle_u64(x1);

    let even = (u1 << 32) | (u0 & 0x00000000FFFFFFFF);
    let odd = (u0 >> 32)  | (u1 & 0xFFFFFFFF00000000);
    (odd, even)
}

pub(crate) fn unshuffle2_u32(x: u32) -> u32 {
    let mut t = x;
    t = bit_permute_step_u32(t, 0x0C0C0C0C, 2);
    t = bit_permute_step_u32(t, 0x22222222, 1);
    t = bit_permute_step_u32(t, 0x0000F0F0, 12);
    t = bit_permute_step_u32(t, 0x00CC00CC, 6);
    t
}


pub(crate) fn unshuffle2_u64(x: u64) -> u64 {
    let mut t = x;
    t = bit_permute_step_u64(t, 0x00CC00CC00CC00CC, 6);
    t = bit_permute_step_u64(t, 0x0A0A0A0A0A0A0A0A, 3);
    t = bit_permute_step_u64(t, 0x00000000FF00FF00, 24);
    t = bit_permute_step_u64(t, 0x0000F0F00000F0F0, 12);
    t
}

/// Transposes the bits in a 64-bit unsigned integer.
/// This function rearranges the bits according to specific masks and shifts,
/// effectively performing a bitwise transpose operation.
pub(crate) fn transpose(x: u64) -> u64 {
    let mut t = x;
    t = bit_permute_step_u64(t, 0x00000000F0F0F0F0, 28);
    t = bit_permute_step_u64(t, 0x0000CCCC0000CCCC, 14);
    t = bit_permute_step_u64(t, 0x00AA00AA00AA00AA, 7);
    t
}