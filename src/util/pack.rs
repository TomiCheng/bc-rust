
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
pub(crate) fn be_to_u32_buffer(bs: &[u8], ns: &mut [u32]) {
    const SIZE: usize = size_of::<u32>();
    let mut n = 0usize;
    for i in 0..ns.len() {
        ns[i] = u32::from_be_bytes(bs[n..(n + SIZE)].try_into().unwrap());
        n += SIZE;
    }
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