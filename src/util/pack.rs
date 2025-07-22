pub(crate) fn be_to_u32_low(bs: &[u8]) -> u32 {
    let len = bs.len();
    debug_assert!(1 <= len && len <= 4);
    let mut result = bs[0] as u32;
    for i in 1..len {
        result <<= 8;
        result |= bs[i] as u32;
    }
    result
}
pub(crate) fn le_to_u32_low(bs: &[u8]) -> u32 {
    let len = bs.len();
    debug_assert!(1 <= len && len <= 4, "Invalid length: {}", len);
    let mut result = bs[0] as u32;
    let mut pos = 0usize;
    for i in 1..len {
        pos += 8;
        result |= (bs[i] as u32) << pos;
    }
    result
}

pub(crate) fn u32_to_be_bytes(n: u32, bs: &mut [u8]) {
    bs[0] = (n >> 24) as u8;
    bs[1] = (n >> 16) as u8;
    bs[2] = (n >> 8) as u8;
    bs[3] = n as u8;
}

pub(crate) fn be_to_u32(bs: &[u8]) -> u32 {
    u32::from_be_bytes(bs.try_into().unwrap())
}
pub(crate) fn le_to_u64(bs: &[u8]) -> u64 {
    u64::from_le_bytes(bs.try_into().unwrap())
}


pub(crate) fn u64_to_le(n: u64, bs: &mut [u8]) {
    bs[0..size_of::<u64>()].copy_from_slice(&n.to_le_bytes())
}
pub(crate) fn u64_to_be(n: u64, bs: &mut [u8]) {
    bs[0..size_of::<u64>()].copy_from_slice(&n.to_be_bytes())
}


// Vec
pub(crate) fn be_to_u32_buffer(bs: &[u8], ns: &mut [u32]) {
    const SIZE: usize = size_of::<u32>();
    let mut n = 0usize;
    for i in 0..ns.len() {
        ns[i] = u32::from_be_bytes(bs[n..(n + SIZE)].try_into().unwrap());
        n += SIZE;
    }
}
pub(crate) fn u32_to_u8_vec_le(input: &[u32], output: &mut [u8]) {
    const SIZE: usize = size_of::<u32>();
    let mut n = 0usize;
    for &i in input {
        output[n..(n + SIZE)].copy_from_slice(&i.to_le_bytes());
        n += SIZE;
    }
}