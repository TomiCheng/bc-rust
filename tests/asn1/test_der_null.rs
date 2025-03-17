use bc_rust::asn1;
use bc_rust::asn1::asn1_encodable;
use bc_rust::asn1::asn1_object;
use bc_rust::asn1::Asn1Object;

#[test]
fn test_display() {
    let null = asn1::DerNull::default();
    assert_eq!("NULL".to_string(), null.to_string());
}
#[test]
fn test_encodable() {
    let null_buffer = vec![0x05, 0x00];
    let null: Box<dyn Asn1Object> = Box::new(asn1::DerNull::default());
    {
        let buffer = null.get_encoded().expect("fail");
        assert_eq!(2, buffer.len());
        assert_eq!(null_buffer, buffer);
    }
    {
        let buffer = null
            .get_encoded_with_encoding(asn1_encodable::DER)
            .expect("fail");
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
            .encode_to_with_encoding(&mut buffer, asn1_encodable::DER)
            .expect("fail");
        assert_eq!(2, length);
        assert_eq!(null_buffer, buffer);
    }
}

#[test]
fn test_parse_asn1_object() {
    let buffer = vec![0x05u8, 0x00];
    let asn1_object = asn1_object::from_read(&mut buffer.as_slice()).expect("fail");
    assert!(asn1_object.as_any().is::<asn1::DerNull>());
}
