use std::rc::Rc;
use bc_rust::asn1::asn1_encodable::DER;
use bc_rust::asn1::{Asn1Convertiable, DerBoolean};

#[test]
fn test_new() {
    let asn1_true = DerBoolean::new(true);
    assert!(asn1_true.is_true());

    let asn1_false = DerBoolean::new(false);
    assert!(!asn1_false.is_true());
}

#[test]
fn test_create_i32() {
    let asn1_true = DerBoolean::with_i32(1);
    assert!(asn1_true.is_true());

    let asn1_false = DerBoolean::with_i32(0);
    assert!(!asn1_false.is_true());
}

#[test]
fn test_display() {
    let asn1_true = DerBoolean::new(true);
    assert_eq!("TRUE", asn1_true.to_string());

    let asn1_false = DerBoolean::new(false);
    assert_eq!("FALSE", asn1_false.to_string());
}

#[test]
fn test_encodable() {
    let result_length = 3;
    let result_buffer = vec![0x01, 0x01, 0xFF];
    let asn1_true = Rc::new(DerBoolean::new(true));
    let asn1_encodable = asn1_true.to_asn1_encodable();
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
