// pub(crate) fn le_to_u32(bs: &[u8]) -> u32 {
//     ((bs[0] as u32) << 0) | ((bs[1] as u32) << 8) | ((bs[2] as u32) << 16) | ((bs[3] as u32) << 24)
// }

// pub(crate) fn u32_to_le(n: u32, bs: &mut[u8]) {
//     bs[0] = (n >> 0) as u8;
//     bs[1] = (n >> 8) as u8;
//     bs[2] = (n >> 16) as u8;
//     bs[3] = (n >> 24) as u8;
// }


pub(crate) fn read_le_u32(input: &mut &[u8]) -> u32 {
    let (int_bytes, rest) = input.split_at(std::mem::size_of::<u32>());
    *input = rest;
    u32::from_le_bytes(int_bytes.try_into().unwrap())
}

pub(crate) fn write_le_u32(input: u32, output: &mut [u8]) -> &mut [u8] {
    let (int_bytes, rest) = output.split_at_mut(std::mem::size_of::<u32>());
    input.to_le_bytes().clone_into(int_bytes.try_into().unwrap());
    rest
}

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
    debug_assert!(1 <= len && len <= 4);
    let mut result = bs[0] as u32;
    let mut pos = 0usize;
    for i in 1..len {
        pos += 8;
        result |= (bs[i] as u32) << pos;
    }
    result
}
#[test]
fn test_le_to_u32_low() {
    let bs = [0x12, 0x34, 0x56];
    assert_eq!(le_to_u32_low(&bs), 0x563412);
}
pub(crate) fn be_to_u32_buffer(bs: &[u8], ns: &mut [u32]) {
    const SIZE: usize = size_of::<u32>();
    let mut n = 0usize;
    for i in 0..ns.len() {
        ns[i] = u32::from_be_bytes(bs[n..(n + SIZE)].try_into().unwrap());
        n += SIZE;
    }
}

pub(crate) fn u32_to_be_bytes(n: u32, bs: &mut [u8]) {
    bs[0] = (n >> 24) as u8;
    bs[1] = (n >> 16) as u8;
    bs[2] = (n >> 8) as u8;
    bs[3] = n as u8;
}