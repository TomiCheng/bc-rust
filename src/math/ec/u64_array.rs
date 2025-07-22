use crate::math::BigInteger;
use crate::{BcError, Result};
use crate::math::big_integer::ZERO;
use crate::math::raw::nat;

pub(crate) struct U64Array {
    data: Vec<u64>,
}

impl U64Array {
    pub(crate) fn new(data: Vec<u64>) -> Self {
        U64Array { data }
    }
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        U64Array {
            data: Vec::with_capacity(capacity),
        }
    }
    pub(crate) fn with_slice(slice: &[u64]) -> Self {
        U64Array {
            data: slice.to_vec(),
        }
    }
    pub(crate) fn with_big_integer(n: &BigInteger) -> Result<Self> {
        if n.sign() < 0 {
            return Err(BcError::with_invalid_argument("invalid F2m field value"))
        }

        if n.sign() == 0 {
            return Ok(U64Array::new(vec![0; 1]));
        }

        let barr = n.to_vec();
        let mut barr_len = barr.len();
        let mut barr_start = 0;
        if barr[0] == 0 {
            barr_len -= 1;
            barr_start = 1;
        }

        let int_len = (barr_len + 7) / 8;
        let mut data = Vec::with_capacity(int_len);

        let mut i_arr_j = (int_len - 1) as isize;
        let rem = barr_len % 8 + barr_start;
        let mut temp = 0u64;
        let mut barr_i = barr_start;
        if barr_start < rem {
            while barr_i < rem {
                temp <<= 8;
                let barr_barr_i = barr[barr_i];
                temp |= barr_barr_i as u64;
                barr_i += 1;
            }
            data[i_arr_j as usize] = temp;
            i_arr_j -= 1;
        }

        while i_arr_j >= 0 {
            temp = 0;
            for _ in 0..8 {
                temp <<= 8;
                let barr_barr_i = barr[barr_i];
                barr_i += 1;
                temp |= barr_barr_i as u64;
            }
            data[i_arr_j as usize] = temp;
        }
        Ok(U64Array { data })
    }
    pub(crate) fn copy_to(&self, z: &mut [u64]) {
        z.copy_from_slice(&self.data)
    }
    pub(crate) fn is_one(&self) -> bool {
        let a = &self.data;
        let a_len = a.len();
        if a_len < 1 || a[0] != 1 {
            return false;
        }

        for i in 1..a_len {
            if a[i] != 0 {
                return false;
            }
        }
        true
    }
    pub(crate) fn is_zero(&self) -> bool {
        let a = &self.data;
        for &value in a {
            if value != 0 {
                return false;
            }
        }
        true
    }
    pub(crate) fn get_used_length(&self) -> usize {
        self.get_used_length_from(self.data.len())
    }
    pub(crate) fn get_used_length_from(&self, from: usize) -> usize {
        let a = &self.data;
        let mut from = from.min(a.len());

        if from < 1 {
            return 0;
        }

        if a[0] != 0 {
            while a[{
                from -= 1;
                from
            }] == 0 {}
            return from + 1;
        }

        loop {
            if a[{
                from -= 1;
                from
            }] != 0 {
                return from + 1;
            }

            if from == 0 {
                break;
            }
        }
        0
    }
    pub(crate) fn degree(&self) -> usize {
        let mut i = self.data.len();
        let mut w;
        loop {
            if i == 0 {
                return 0;
            }
            w = self.data[{
                i -= 1;
                i
            }];

            if w != 0 {
                break;
            }
        }
        (i << 6) + Self::bit_length(w)
    }
    fn degree_from(&self, limit: usize) -> usize {
        let mut i = (limit + 62) >> 6;
        let mut w;
        loop {
            if i == 0 {
                return 0;
            }
            w = self.data[{
                i -= 1;
                i
            }];

            if w != 0 {
                break;
            }
        }
        (i << 6) + Self::bit_length(w)
    }
    fn bit_length(w: u64) -> usize {
        64 - (w as i64).leading_zeros() as usize
    }
    fn resized_data(&self, new_len: usize) -> Vec<u64> {
        let len = self.data.len().min(new_len);
        let mut new_ints = vec![0; new_len];
        new_ints[0..len].copy_from_slice(&self.data[0..len]);
        new_ints
    }
    pub(crate) fn to_big_integer(&self) -> BigInteger {
        let used_len = self.get_used_length();
        if used_len == 0 {
            return (*ZERO).clone();
        }
        let highest_int = self.data[used_len - 1];
        let mut temp = vec![0u8; 8];
        let mut barr_i = 0;
        let mut trailing_zero_bytes_done = false;
        for j in (0..8).rev() {
            let this_byte = (highest_int >> (8 * j)) as u8;
            if trailing_zero_bytes_done || this_byte != 0 {
                trailing_zero_bytes_done = true;
                temp[barr_i] = this_byte;
            }
        }

        let barr_len = 8 * (used_len -1) + barr_i;
        let mut barr = vec![0u8; barr_len];
        for j in 0..barr_i {
            barr[j] = temp[j];
        }

        // Highest value int is done now
        for i_arr_j in (0..(used_len - 2)).rev() {
            let mi = self.data[i_arr_j];
            for j in (0..8).rev() {
                barr[barr_i] = (mi >> (8 * j)) as u8;
                barr_i += 1;
            }
        }

        BigInteger::with_sign_buffer(1, &barr).unwrap()
    }
    fn shift_up_self(x: &mut [u64], shift: usize)-> u64 {
        let shift_inv = 64 - shift;
        let mut prev = 0u64;
        for i in 0..x.len() {
            let next = x[i];
            x[i] = (next << shift_inv) | prev;
            prev = next >> shift_inv;
        }
        prev
    }
    fn shift_up(from: &[u64], to: &mut [u64], shift: usize) -> u64 {
        let shift_inv = 64 - shift;
        let mut prev = 0u64;
        for i in 0..to.len() {
            let next = from[i];
            to[i] = (next << shift_inv) | prev;
            prev = next >> shift_inv;
        }
        prev
    }
    pub(crate) fn add_one(&self) -> U64Array {
        if self.data.len() == 0 {
            return U64Array::new(vec![1u64; 1]);
        }

        let result_len = 1.max(self.get_used_length());
        let mut data = self.resized_data(result_len);
        data[0] ^= 1;
        Self::new(data)
    }
    fn add_shifted_by_bits_safe() {
        todo!();
    }

    fn add(x: &[u64]) {
    }
}