use bc_rust::asn1;
use bc_rust::asn1::asn1_encodable;
use bc_rust::asn1::asn1_object;
use bc_rust::asn1::Asn1Encodable;

#[test]
fn test_new() {
    {
        let der_boolean = asn1::DerBoolean::new(false);
        assert!(!der_boolean.is_true());
    }
    {
        let der_boolean = asn1::DerBoolean::new(true);
        assert!(der_boolean.is_true());
    }
}

#[test]
fn test_create_i32() {
    {
        let der_boolean = asn1::DerBoolean::with_i32(1);
        assert!(der_boolean.is_true());
    }
    {
        let der_boolean = asn1::DerBoolean::with_i32(0);
        assert!(!der_boolean.is_true());
    }
}

#[test]
fn test_display() {
    {
        let asn1_object = asn1::DerBoolean::new(true);
        assert_eq!("TRUE", asn1_object.to_string());
    }
    {
        let asn1_object = asn1::DerBoolean::new(false);
        assert_eq!("FALSE", asn1_object.to_string());
    }
}

#[test]
fn test_encodable() {
    let result_length = 3;
    let result_buffer = vec![0x01, 0x01, 0xFF];
    let asn1_object = asn1::DerBoolean::new(true);
    {
        let buffer = asn1_object.get_encoded().expect("fail");
        assert_eq!(result_length, buffer.len());
        assert_eq!(result_buffer, buffer);
    }
    {
        let buffer = asn1_object
            .get_encoded_with_encoding(asn1_encodable::DER)
            .expect("fail");
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
            .encode_to_with_encoding(&mut buffer, asn1_encodable::DER)
            .expect("fail");
        assert_eq!(result_length, length);
        assert_eq!(result_buffer, buffer);
    }
}

#[test]
fn test_parse_asn1_object() {
    let buffer = vec![0x01, 0x01, 0xFF];
    let asn1_object = asn1_object::from_read(&mut buffer.as_slice()).expect("fail");
    assert!(asn1_object.as_any().is::<asn1::DerBoolean>());
    let binding = asn1_object.as_any();
    let der_boolean = binding.downcast_ref::<asn1::DerBoolean>().unwrap();
    assert!(der_boolean.is_true());
}
