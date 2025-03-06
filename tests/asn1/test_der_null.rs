use bc_rust::asn1::asn1_encodable::DER;
use bc_rust::asn1::Asn1Encodable;
use bc_rust::asn1::Asn1Object;

#[test]
fn test_display() {
    let null = Asn1Object::with_null();
    assert_eq!("NULL".to_string(), null.to_string());
}
#[test]
fn test_encodable() {
    let null_buffer = vec![0x05, 0x00];
    let null = Asn1Object::with_null();
    {
         let buffer = null.get_encoded().expect("fail");
         assert_eq!(2, buffer.len());
         assert_eq!(null_buffer, buffer);
    }
    {
        let buffer = null.get_encoded_with_encoding(DER).expect("fail");
        assert_eq!(2, buffer.len());
        assert_eq!(null_buffer, buffer);
    }
    {
        let mut buffer = Vec::<u8>::new();
        let length = null.encode_to(&mut buffer).expect("fail");
        assert_eq!(2, length);
        assert_eq!(null_buffer, buffer);
    }
    {
        let mut buffer = Vec::<u8>::new();
        let length = null
            .encode_to_with_encoding(&mut buffer, DER)
            .expect("fail");
        assert_eq!(2, length);
        assert_eq!(null_buffer, buffer);
    }
}

#[test]
fn test_parse_asn1_object() {
    let buffer = vec![0x05u8, 0x00];
    let asn1_object = Asn1Object::parse(&mut buffer.as_slice()).expect("fail");
    assert!(asn1_object.is_der_null());
}
