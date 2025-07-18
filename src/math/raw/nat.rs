
pub fn get_bit_length(len: usize, x: &[u32]) -> usize {
    for i in (0..len).rev() {
        let x_i = x[i];
        if x_i != 0 {
            return (i * 32) + 32 - x_i.leading_zeros() as usize;
        }
    }
    0
}

// TODO

pub(crate) fn gte(len: usize, x: &[u32], y: &[u32]) -> bool {
    for i in (0..len).rev() {
        let x_i = x[i];
        let y_i = y[i];
        if x_i < y_i {
            return false;
        }
        if x_i > y_i {
            return true;
        }
    }
    true
}

// TODO

pub(crate) fn less_than(len: usize, x: &[u32], y: &[u32]) -> i32 {
    let mut c = 0;
    for i in 0..len {
        c += x[i] as i64 - y[i] as i64;
        c >>= 32;
    }

    debug_assert!(c == 0 || c == -1);
    c as i32
}