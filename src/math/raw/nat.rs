#![allow(dead_code)]

#[derive(Debug, Clone)]
pub struct Nat<T, const N: usize> {
    data: [T; N],
}

impl<const N: usize> Nat<u32, N> {
    const SIZE: usize = size_of::<u32>();
    const BIT_SIZE: usize = size_of::<u32>() * 8;
    pub fn create() -> Self {
        Self { data: [0u32; N] }
    }

    // Ops
    pub fn add(x: &Self, y: &Self, z: &mut Self) -> u32 {
        let mut c = 0u64;
        for i in 0..N {
            c += x.data[i] as u64 + y.data[i] as u64;
            z.data[i] = c as u32;
            c >>= 32;
        }
        c as u32
    }
}

impl<const N: usize> Nat<u64, N> {
    const SIZE: usize = size_of::<u64>();
    const BIT_SIZE: usize = size_of::<u64>() * 8;
    pub fn create() -> Self {
        Self { data: [0u64; N] }
    }

    // Ops
}

pub type Nat128 = Nat<u32, 4>;
pub type Nat160 = Nat<u32, 5>;
pub type Nat192 = Nat<u32, 6>;
pub type Nat224 = Nat<u32, 7>;
pub type Nat256 = Nat<u32, 8>;
pub type Nat320 = Nat<u64, 5>;
pub type Nat384 = Nat<u64, 6>;
pub type Nat448 = Nat<u64, 7>;
pub type Nat512 = Nat<u64, 8>;
pub type Nat576 = Nat<u64, 9>;