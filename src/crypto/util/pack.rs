pub trait Pack {
    fn from_be_slice(bytes: &[u8]) -> Self;
    fn from_be_slice_low(bytes: &[u8]) -> Self;
    fn from_be_slice_high(bytes: &[u8]) -> Self;

    fn to_be_slice(self, bytes: &mut [u8]);
    fn to_be_vec(self) -> Vec<u8> where Self: Sized {
        let mut v = vec![0u8; size_of::<Self>()];
        self.to_be_slice(&mut v);
        v
    }
    fn to_be_slice_low(self, bytes: &mut [u8]);
    fn to_be_slice_high(self, bytes: &mut [u8]);

    fn from_le_slice(bytes: &[u8]) -> Self;
    fn from_le_slice_low(bytes: &[u8]) -> Self;
    fn from_le_slice_high(bytes: &[u8]) -> Self;

    fn to_le_slice(self, bytes: &mut [u8]);
    fn to_le_vec(self) -> Vec<u8> where Self: Sized {
        let mut v = vec![0u8; size_of::<Self>()];
        self.to_le_slice(&mut v);
        v
    }
    fn to_le_slice_low(self, bytes: &mut [u8]);
    fn to_le_slice_high(self, bytes: &mut [u8]);
}
pub trait FromPacks {
    fn fill_from_be_slice(&mut self, bytes: &[u8]);
    fn fill_from_le_slice(&mut self, bytes: &[u8]);
}
pub trait ToPacks {
    fn fill_to_be_slice(&self, bytes: &mut [u8]);
    //fn to_be_vec(&self) -> Vec<u8> where Self: Sized;
    fn fill_to_le_slice(&self, bytes: &mut [u8]);
    //fn to_le_vec(&self) -> Vec<u8> where Self: Sized;
}
macro_rules! impl_pack_for_uint {
    ($t:ty) => {
        impl Pack for $t {
            #[inline]
            fn from_be_slice(bytes: &[u8]) -> Self {
                Self::from_be_bytes(bytes.try_into().unwrap())
            }
            fn from_be_slice_low(bytes: &[u8]) -> Self {
                const LEN: usize = size_of::<$t>();
                let mut array = [0u8; LEN];
                let len = LEN.min(bytes.len());
                array[LEN - len..].copy_from_slice(&bytes[..len]);
                <$t>::from_be_bytes(array)
            }
            fn from_be_slice_high(bytes: &[u8]) -> Self {
                const LEN: usize = size_of::<$t>();
                let mut array = [0u8; LEN];
                let len = LEN.min(bytes.len());
                if len > 0 {
                    array[..len].copy_from_slice(&bytes[..len]);
                }
                <$t>::from_be_bytes(array)
            }

            fn to_be_slice(self, bytes: &mut [u8]) {
                let buffer = <$t>::to_be_bytes(self);
                bytes[..buffer.len()].copy_from_slice(&buffer);
            }
            fn to_be_slice_low(self, bytes: &mut [u8]) {
                const LEN: usize = size_of::<$t>();
                let buffer = <$t>::to_be_bytes(self);
                let len = buffer.len().min(bytes.len());
                bytes[..len].copy_from_slice(&buffer[LEN - len..]);
            }
            fn to_be_slice_high(self, bytes: &mut [u8]) {
                let buffer = <$t>::to_be_bytes(self);
                let len = buffer.len().min(bytes.len());
                bytes[..len].copy_from_slice(&buffer[..len]);
            }

            fn from_le_slice(bytes: &[u8]) -> Self {
                Self::from_le_bytes(bytes.try_into().unwrap())
            }
            fn from_le_slice_low(bytes: &[u8]) -> Self {
                const LEN: usize = size_of::<$t>();
                let mut array = [0u8; LEN];
                let len = LEN.min(bytes.len());
                array[..len].copy_from_slice(&bytes[..len]);
                <$t>::from_le_bytes(array)
            }
            fn from_le_slice_high(bytes: &[u8]) -> Self {
                const LEN: usize = size_of::<$t>();
                let mut array = [0u8; LEN];
                let len = LEN.min(bytes.len());
                if len > 0 {
                    array[LEN - len..].copy_from_slice(&bytes[..len]);
                }
                <$t>::from_le_bytes(array)
            }

            fn to_le_slice(self, bytes: &mut [u8]) {
                let buffer = <$t>::to_le_bytes(self);
                bytes[..buffer.len()].copy_from_slice(&buffer);
            }
            fn to_le_slice_low(self, bytes: &mut [u8]) {
                let buffer = <$t>::to_le_bytes(self);
                let len = buffer.len().min(bytes.len());
                bytes[..len].copy_from_slice(&buffer[..len]);
            }
            fn to_le_slice_high(self, bytes: &mut [u8]) {
                const LEN: usize = size_of::<$t>();
                let buffer = <$t>::to_le_bytes(self);
                let len = buffer.len().min(bytes.len());
                if len > 0 {
                    bytes[..len].copy_from_slice(&buffer[LEN - len..]);
                }
            }
        }
        impl FromPacks for [$t] {
            fn fill_from_be_slice(&mut self, bytes: &[u8]) {
                const LEN: usize = size_of::<$t>();
                for (i, chunk) in bytes.chunks(LEN).enumerate() {
                    self[i] = <$t>::from_be_slice(chunk);
                }
            }
            fn fill_from_le_slice(&mut self, bytes: &[u8]) {
                const LEN: usize = size_of::<$t>();
                for (i, chunk) in bytes.chunks(LEN).enumerate() {
                    self[i] = <$t>::from_le_slice(chunk);
                }
            }
        }
        impl ToPacks for [$t] {
            fn fill_to_be_slice(&self, bytes: &mut [u8]) {
                const LEN: usize = size_of::<$t>();
                for (i, &value) in self.iter().enumerate() {
                    let start = i * LEN;
                    let end = start + LEN;
                    value.to_be_slice(&mut bytes[start..end]);
                }
            }
            // fn to_be_vec(self) -> Vec<u8> where Self: Sized {
            //     const LEN: usize = size_of::<$t>();
            //     let mut v = vec![0u8; self.len() * LEN];
            //     self.fill_to_be_slice(&mut v);
            //     v
            // }
            fn fill_to_le_slice(&self, bytes: &mut [u8]) {
                const LEN: usize = size_of::<$t>();
                for (i, &value) in self.iter().enumerate() {
                    let start = i * LEN;
                    let end = start + LEN;
                    value.to_le_slice(&mut bytes[start..end]);
                }
            }
            // fn to_le_vec(self) -> Vec<u8> where Self: Sized {
            //     const LEN: usize = size_of::<$t>();
            //     let mut v = vec![0u8; self.len() * LEN];
            //     self.fill_to_le_slice(&mut v);
            //     v
            // }
        }
    };
}

impl_pack_for_uint!(u16);
impl_pack_for_uint!(u32);
impl_pack_for_uint!(u64);
impl_pack_for_uint!(u128);

#[cfg(test)]
mod tests {
    use super::Pack;
    use super::FromPacks;
    use super::ToPacks;

    #[test]
    fn test_pack_be_u32() {
        assert_eq!(0x12345678, u32::from_be_slice(&[0x12, 0x34, 0x56, 0x78]));

        assert_eq!(0x12345678, u32::from_be_slice_low(&[0x12, 0x34, 0x56, 0x78]));
        assert_eq!(0x00123456, u32::from_be_slice_low(&[0x12, 0x34, 0x56]));
        assert_eq!(0x00001234, u32::from_be_slice_low(&[0x12, 0x34]));
        assert_eq!(0x00000012, u32::from_be_slice_low(&[0x12]));

        assert_eq!(0x12345678, u32::from_be_slice_high(&[0x12, 0x34, 0x56, 0x78]));
        assert_eq!(0x12345600, u32::from_be_slice_high(&[0x12, 0x34, 0x56]));
        assert_eq!(0x12340000, u32::from_be_slice_high(&[0x12, 0x34]));
        assert_eq!(0x12000000, u32::from_be_slice_high(&[0x12]));

        let value = 0x12345678u32;
        let mut buffer = [0u8; 5];
        value.to_be_slice(&mut buffer[0..4]);
        assert_eq!(buffer, [0x12, 0x34, 0x56, 0x78, 0x00]);
        assert_eq!(value.to_be_vec(), vec![0x12, 0x34, 0x56, 0x78]);


        let mut buffer = [0u8; 5];
        value.to_be_slice_high(&mut buffer[0..0]);
        assert_eq!(&buffer, &[0x00, 0x00, 0x00, 0x00, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_be_slice_high(&mut buffer[0..1]);
        assert_eq!(&buffer, &[0x12, 0x00, 0x00, 0x00, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_be_slice_high(&mut buffer[0..2]);
        assert_eq!(&buffer, &[0x12, 0x34, 0x00, 0x00, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_be_slice_high(&mut buffer[0..3]);
        assert_eq!(&buffer, &[0x12, 0x34, 0x56, 0x00, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_be_slice_high(&mut buffer[0..4]);
        assert_eq!(&buffer, &[0x12, 0x34, 0x56, 0x78, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_be_slice_high(&mut buffer[0..5]);
        assert_eq!(&buffer, &[0x12, 0x34, 0x56, 0x78, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_be_slice_low(&mut buffer[0..0]);
        assert_eq!(&buffer, &[0x00, 0x00, 0x00, 0x00, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_be_slice_low(&mut buffer[0..1]);
        assert_eq!(&buffer, &[0x78, 0x00, 0x00, 0x00, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_be_slice_low(&mut buffer[0..2]);
        assert_eq!(&buffer, &[0x56, 0x78, 0x00, 0x00, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_be_slice_low(&mut buffer[0..3]);
        assert_eq!(&buffer, &[0x34, 0x56, 0x78, 0x00, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_be_slice_low(&mut buffer[0..4]);
        assert_eq!(&buffer, &[0x12, 0x34, 0x56, 0x78, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_be_slice_low(&mut buffer[0..5]);
        assert_eq!(&buffer, &[0x12, 0x34, 0x56, 0x78, 0x00]);

    }
    #[test]
    fn test_pack_le_u32() {
        assert_eq!(0x12345678, u32::from_le_slice(&[0x78, 0x56, 0x34, 0x12]));

        assert_eq!(0x12345678, u32::from_le_slice_low(&[0x78, 0x56, 0x34, 0x12]));
        assert_eq!(0x00345678, u32::from_le_slice_low(&[0x78, 0x56, 0x34]));
        assert_eq!(0x00005678, u32::from_le_slice_low(&[0x78, 0x56]));
        assert_eq!(0x00000078, u32::from_le_slice_low(&[0x78]));

        assert_eq!(0x12345678, u32::from_le_slice_high(&[0x78, 0x56, 0x34, 0x12]));
        assert_eq!(0x12345600, u32::from_le_slice_high(&[0x56, 0x34, 0x12]));
        assert_eq!(0x12340000, u32::from_le_slice_high(&[0x34, 0x12]));
        assert_eq!(0x12000000, u32::from_le_slice_high(&[0x12]));

        let value = 0x12345678u32;
        let mut buffer = [0u8; 5];
        value.to_le_slice(&mut buffer[0..4]);
        assert_eq!(buffer, [0x78, 0x56, 0x34, 0x12, 0x00]);
        assert_eq!(value.to_le_vec(), vec![0x78, 0x56, 0x34, 0x12]);


        let mut buffer = [0u8; 5];
        value.to_le_slice_high(&mut buffer[0..0]);
        assert_eq!(&buffer, &[0x00, 0x00, 0x00, 0x00, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_le_slice_high(&mut buffer[0..1]);
        assert_eq!(&buffer, &[0x12, 0x00, 0x00, 0x00, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_le_slice_high(&mut buffer[0..2]);
        assert_eq!(&buffer, &[0x34, 0x12, 0x00, 0x00, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_le_slice_high(&mut buffer[0..3]);
        assert_eq!(&buffer, &[0x56, 0x34, 0x12, 0x00, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_le_slice_high(&mut buffer[0..4]);
        assert_eq!(&buffer, &[0x78, 0x56, 0x34, 0x12, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_le_slice_high(&mut buffer[0..5]);
        assert_eq!(&buffer, &[0x78, 0x56, 0x34, 0x12, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_le_slice_low(&mut buffer[0..0]);
        assert_eq!(&buffer, &[0x00, 0x00, 0x00, 0x00, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_le_slice_low(&mut buffer[0..1]);
        assert_eq!(&buffer, &[0x78, 0x00, 0x00, 0x00, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_le_slice_low(&mut buffer[0..2]);
        assert_eq!(&buffer, &[0x78, 0x56, 0x00, 0x00, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_le_slice_low(&mut buffer[0..3]);
        assert_eq!(&buffer, &[0x78, 0x56, 0x34, 0x00, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_le_slice_low(&mut buffer[0..4]);
        assert_eq!(&buffer, &[0x78, 0x56, 0x34, 0x12, 0x00]);

        let mut buffer = [0u8; 5];
        value.to_le_slice_low(&mut buffer[0..5]);
        assert_eq!(&buffer, &[0x78, 0x56, 0x34, 0x12, 0x00]);
    }


    #[test]
    fn test_from_packs_be_u16() {
        let bytes = [0x12, 0x34, 0x56, 0x78];
        let mut array = [0u16; 2];
        array.as_mut().fill_from_be_slice(&bytes);
        assert_eq!(array, [0x1234, 0x5678]);
    }
    #[test]
    fn test_to_packs_be_u16() {
        let array = [0x1234u16, 0x5678u16];
        let mut bytes = [0u8; 4];
        array.as_ref().fill_to_be_slice(&mut bytes);
        assert_eq!(bytes, [0x12, 0x34, 0x56, 0x78]);
        //assert_eq!(array.as_ref().to_be_vec(), vec![0x12, 0x34, 0x56, 0x78]);
    }

    #[test]
    fn test_from_packs_le_u16() {
        let bytes = [0x12, 0x34, 0x56, 0x78];
        let mut array = [0u16; 2];
        array.as_mut().fill_from_le_slice(&bytes);
        assert_eq!(array, [0x3412, 0x7856]);
    }
    #[test]
    fn test_to_packs_le_u16() {
        let array = [0x1234u16, 0x5678u16];
        let mut bytes = [0u8; 4];
        array.as_ref().fill_to_le_slice(&mut bytes);
        assert_eq!(bytes, [0x34, 0x12, 0x78, 0x56]);
        //assert_eq!(array.as_ref().to_le_vec(), vec![0x34, 0x12, 0x78, 0x56]);
    }
}