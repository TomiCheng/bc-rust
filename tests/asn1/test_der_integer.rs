use bc_rust::asn1::asn1_encodable::DER;
use bc_rust::asn1::{Asn1Convertiable, DerInteger};
use bc_rust::math::BigInteger;
use bc_rust::util::encoders::hex::to_decode_with_str;
use std::rc::Rc;

fn check_i32_value(i: &DerInteger, n: i32) {
    let val = i.get_value();
    assert_eq!(val.get_i32_value(), n);
    assert_eq!(val.try_get_i32_value(), Some(n));
    assert_eq!(i.try_get_i32_value(), Some(n));
    assert!(i.has_i32_value(n));
}

fn check_i64_value(i: &DerInteger, n: i64) {
    let val = i.get_value();
    assert_eq!(val.get_i64_value(), n);
    assert_eq!(val.try_get_i64_value(), Some(n));
    assert_eq!(i.try_get_i64_value(), Some(n));
    assert!(i.has_i64_value(n));
}
/// Ensure existing single byte behavior.
#[test]
fn test_valid_encoding_single_byte() {
    let raw_i32 = vec![0x10];
    let i = DerInteger::with_buffer(&raw_i32).expect("error");
    check_i32_value(&i, 16);
}

#[test]
fn test_valid_encoding_multi_byte() {
    let raw_i32 = vec![0x10, 0xFF];
    let i = DerInteger::with_buffer(&raw_i32).expect("error");
    check_i32_value(&i, 4351);
}

#[test]
fn test_invalid_encoding_00() {
    let raw_i32 = vec![0x00, 0x10, 0xFF];
    let i = DerInteger::with_buffer(&raw_i32);
    assert!(i.is_err());
}

#[test]
fn test_invalid_encoding_ff() {
    let raw_i32 = vec![0xFF, 0x81, 0xFF];
    let i = DerInteger::with_buffer(&raw_i32);
    assert!(i.is_err());
}

#[test]
fn test_invalid_encoding_00_32bits() {
    // Check what would pass loose validation fails outside of loose validation.
    let raw_i32 = vec![0x00, 0x00, 0x00, 0x00, 0x10, 0xFF];
    let i = DerInteger::with_buffer(&raw_i32);
    assert!(i.is_err());
}

#[test]
fn test_invalid_encoding_ff_32bits() {
    // Check what would pass loose validation fails outside of loose validation.
    let raw_i32 = vec![0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0xFF];
    let i = DerInteger::with_buffer(&raw_i32);
    assert!(i.is_err());
}

#[test]
fn test_loose_valid_encoding_zero_32b_aligned() {
    let raw_i64 = to_decode_with_str("00000010FF000000").unwrap();
    let i = DerInteger::with_buffer_allow_unsafe(&raw_i64).expect("error");
    check_i64_value(&i, 72997666816);
}

#[test]
fn test_loose_valid_encoding_ff_32b_aligned() {
    let raw_i64 = to_decode_with_str("FFFFFF10FF000000").unwrap();
    let i = DerInteger::with_buffer_allow_unsafe(&raw_i64).expect("error");
    check_i64_value(&i, -1026513960960);
}

#[test]
fn test_loose_valid_encoding_ff_32b_aligned_1not0() {
    let raw_i64 = to_decode_with_str("FFFEFF10FF000000").unwrap();
    let i = DerInteger::with_buffer_allow_unsafe(&raw_i64).expect("error");
    check_i64_value(&i, -282501490671616);
}

#[test]
fn test_loose_valid_encoding_ff_32b_aligned_2not0() {
    let raw_i64 = to_decode_with_str("FFFFFE10FF000000").unwrap();
    let i = DerInteger::with_buffer_allow_unsafe(&raw_i64).expect("error");
    check_i64_value(&i, -2126025588736);
}

#[test]
fn test_oversized_encoding() {
    // Should pass as loose validation permits 3 leading 0xFF bytes.
    let der_integer = DerInteger::with_buffer_allow_unsafe(
        &to_decode_with_str("FFFFFFFE10FF000000000000").unwrap(),
    )
    .unwrap();
    let big_integer =
        BigInteger::with_buffer(&to_decode_with_str("FFFFFFFE10FF000000000000").unwrap());

    assert_eq!(der_integer.get_value(), big_integer);
}

#[test]
fn test_encode() {
    let result_length = 6;
    let result_buffer = vec![0x02, 0x04, 0x07, 0x5B, 0xCD, 0x15];
    let asn1_integer = Rc::new(DerInteger::with_i64(123456789));
    let asn1_encodable = asn1_integer.to_asn1_encodable();
    {
        let buffer = asn1_encodable.get_encoded().expect("fail");
        assert_eq!(result_length, buffer.len());
        assert_eq!(result_buffer, buffer);
    }
    {
        let buffer = asn1_encodable.get_encoded_with_encoding(DER).expect("fail");
        assert_eq!(result_length, buffer.len());
        assert_eq!(result_buffer, buffer);
    }
    {
        let mut buffer = Vec::<u8>::new();
        let length = asn1_encodable.encode_to(&mut buffer).expect("fail");
        assert_eq!(result_length, length);
        assert_eq!(result_buffer, buffer);
    }
    {
        let mut buffer = Vec::<u8>::new();
        let length = asn1_encodable
            .encode_to_with_encoding(&mut buffer, DER)
            .expect("fail");
        assert_eq!(result_length, length);
        assert_eq!(result_buffer, buffer);
    }
}
