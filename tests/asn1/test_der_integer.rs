use bc_rust::asn1;
use bc_rust::asn1::asn1_encodable;
use bc_rust::asn1::asn1_object;
use bc_rust::asn1::Asn1Object;
use bc_rust::math;
use bc_rust::util::encoders::hex;

fn check_i32_value(obj: &asn1::DerInteger, n: i32) {
    let val = obj.get_value();
    assert_eq!(val.i32_value(), n);
    assert_eq!(val.try_get_i32_value(), Some(n));
    assert_eq!(obj.try_get_i32_value(), Some(n));
    assert!(obj.has_i32_value(n));
}

fn check_i64_value(obj: &asn1::DerInteger, n: i64) {
    let val = obj.get_value();
    assert_eq!(val.get_i64_value(), n);
    assert_eq!(val.try_get_i64_value(), Some(n));
    assert_eq!(obj.try_get_i64_value(), Some(n));
    assert!(obj.has_i64_value(n));
}

/// Ensure existing single byte behavior.
#[test]
fn test_valid_encoding_single_byte() {
    let raw_i32 = vec![0x10];
    let i = asn1::DerInteger::with_buffer(&raw_i32).expect("error");
    check_i32_value(&i, 16);
}

#[test]
fn test_valid_encoding_multi_byte() {
    let raw_i32 = vec![0x10, 0xFF];
    let i = asn1::DerInteger::with_buffer(&raw_i32)
        .expect("error")
        .into();
    check_i32_value(&i, 4351);
}

#[test]
fn test_invalid_encoding_00() {
    let raw_i32 = vec![0x00, 0x10, 0xFF];
    let i = asn1::DerInteger::with_buffer(&raw_i32);
    assert!(i.is_err());
}

#[test]
fn test_invalid_encoding_ff() {
    let raw_i32 = vec![0xFF, 0x81, 0xFF];
    let i = asn1::DerInteger::with_buffer(&raw_i32);
    assert!(i.is_err());
}

#[test]
fn test_invalid_encoding_00_32bits() {
    // Check what would pass loose validation fails outside of loose validation.
    let raw_i32 = vec![0x00, 0x00, 0x00, 0x00, 0x10, 0xFF];
    let i = asn1::DerInteger::with_buffer(&raw_i32);
    assert!(i.is_err());
}

#[test]
fn test_invalid_encoding_ff_32bits() {
    // Check what would pass loose validation fails outside of loose validation.
    let raw_i32 = vec![0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0xFF];
    let i = asn1::DerInteger::with_buffer(&raw_i32);
    assert!(i.is_err());
}

#[test]
fn test_loose_valid_encoding_zero_32b_aligned() {
    let raw_i64 = hex::to_decode_with_str("00000010FF000000").unwrap();
    let i = asn1::DerInteger::with_buffer_allow_unsafe(&raw_i64)
        .expect("error")
        .into();
    check_i64_value(&i, 72997666816);
}

#[test]
fn test_loose_valid_encoding_ff_32b_aligned() {
    let raw_i64 = hex::to_decode_with_str("FFFFFF10FF000000").unwrap();
    let i = asn1::DerInteger::with_buffer_allow_unsafe(&raw_i64)
        .expect("error")
        .into();
    check_i64_value(&i, -1026513960960);
}

#[test]
fn test_loose_valid_encoding_ff_32b_aligned_1not0() {
    let raw_i64 = hex::to_decode_with_str("FFFEFF10FF000000").unwrap();
    let i = asn1::DerInteger::with_buffer_allow_unsafe(&raw_i64)
        .expect("error")
        .into();
    check_i64_value(&i, -282501490671616);
}

#[test]
fn test_loose_valid_encoding_ff_32b_aligned_2not0() {
    let raw_i64 = hex::to_decode_with_str("FFFFFE10FF000000").unwrap();
    let i = asn1::DerInteger::with_buffer_allow_unsafe(&raw_i64)
        .expect("error")
        .into();
    check_i64_value(&i, -2126025588736);
}

#[test]
fn test_oversized_encoding() {
    // Should pass as loose validation permits 3 leading 0xFF bytes.
    let der_integer = asn1::DerInteger::with_buffer_allow_unsafe(
        &hex::to_decode_with_str("FFFFFFFE10FF000000000000").unwrap(),
    )
    .unwrap();
    let big_integer = math::BigInteger::with_buffer(
        &hex::to_decode_with_str("FFFFFFFE10FF000000000000").unwrap(),
    );

    assert_eq!(der_integer.get_value(), big_integer);
}

#[test]
fn test_encode() {
    let result_length = 6;
    let result_buffer = vec![0x02, 0x04, 0x07, 0x5B, 0xCD, 0x15];
    let asn1_integer: Box<dyn Asn1Object> = Box::new(asn1::DerInteger::with_i64(123456789));
    {
        let buffer = asn1_integer.get_encoded().expect("fail");
        assert_eq!(result_length, buffer.len());
        assert_eq!(result_buffer, buffer);
    }
    {
        let buffer = asn1_integer.get_encoded_with_encoding(asn1_encodable::DER).expect("fail");
        assert_eq!(result_length, buffer.len());
        assert_eq!(result_buffer, buffer);
    }
    {
        let mut buffer = Vec::<u8>::new();
        let length = asn1_integer.encode_to(&mut buffer).expect("fail");
        assert_eq!(result_length, length);
        assert_eq!(result_buffer, buffer);
    }
    {
        let mut buffer = Vec::<u8>::new();
        let length = asn1_integer
            .encode_to_with_encoding(&mut buffer, asn1_encodable::DER)
            .expect("fail");
        assert_eq!(result_length, length);
        assert_eq!(result_buffer, buffer);
    }
}

#[test]
fn test_parse_asn1_object() {
    let buffer = vec![0x02u8, 0x04, 0x07, 0x5B, 0xCD, 0x15];
    let asn1_object = asn1_object::from_read(&mut buffer.as_slice()).expect("fail");
    assert!(asn1_object.as_any().is::<asn1::DerInteger>());
    check_i64_value(
        &asn1_object
            .as_any()
            .downcast_ref::<asn1::DerInteger>()
            .unwrap(),
        123456789,
    );
}
