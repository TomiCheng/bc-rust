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