use bc_rust::asn1::{Asn1Encodable, Asn1Object, DerBooleanImpl};
use bc_rust::asn1::asn1_encodable::DER;

#[test]
fn test_new() {
    {
        let asn1_object = Asn1Object::with_bool(false);
        let der_boolean = asn1_object.as_der_boolean();
        assert!(!der_boolean.is_true());
    }
    {
        let asn1_object = Asn1Object::with_bool(true);
        let der_boolean = asn1_object.as_der_boolean();
        assert!(der_boolean.is_true());
    }
}

#[test]
fn test_create_i32() {
    {
        let asn1_object = Asn1Object::new_der_boolean(DerBooleanImpl::with_i32(1));
        let der_boolean = asn1_object.as_der_boolean();
        assert!(der_boolean.is_true());
    }
    {
        let asn1_object = Asn1Object::new_der_boolean(DerBooleanImpl::with_i32(0));
        let der_boolean = asn1_object.as_der_boolean();
        assert!(!der_boolean.is_true());
    }
}

#[test]
fn test_display() {
    {
        let asn1_object = Asn1Object::with_bool(true);
        assert_eq!("TRUE", asn1_object.to_string());
    }
    {
        let asn1_object = Asn1Object::with_bool(false);
        assert_eq!("FALSE", asn1_object.to_string());
    }
}

#[test]
fn test_encodable() {
    let result_length = 3;
    let result_buffer = vec![0x01, 0x01, 0xFF];
    let asn1_object = Asn1Object::with_bool(true);
    {
        let buffer = asn1_object.get_encoded().expect("fail");
        assert_eq!(result_length, buffer.len());
        assert_eq!(result_buffer, buffer);
    }
    {
        let buffer = asn1_object.get_encoded_with_encoding(DER).expect("fail");
        assert_eq!(result_length, buffer.len());
        assert_eq!(result_buffer, buffer);
    }
    {
        let mut buffer = Vec::<u8>::new();
        let length = asn1_object.encode_to(&mut buffer).expect("fail");
        assert_eq!(result_length, length);
        assert_eq!(result_buffer, buffer);
    }
    {
        let mut buffer = Vec::<u8>::new();
        let length = asn1_object
            .encode_to_with_encoding(&mut buffer, DER)
            .expect("fail");
        assert_eq!(result_length, length);
        assert_eq!(result_buffer, buffer);
    }
}

#[test]
fn test_parse_asn1_object() {
    let buffer = vec![0x01, 0x01, 0xFF];
    let asn1_object = Asn1Object::parse(&mut buffer.as_slice()).expect("fail");
    assert!(asn1_object.is_der_boolean());
    let der_boolean = asn1_object.as_der_boolean();
    assert!(der_boolean.is_true());
}

