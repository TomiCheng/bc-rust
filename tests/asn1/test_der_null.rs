use bc_rust::asn1::asn1_encodable::DER;
use bc_rust::asn1::{Asn1Convertiable, DerNull};
use std::rc::Rc;

#[test]
fn test_display() {
    let null = DerNull::new();
    assert_eq!("NULL".to_string(), null.to_string());
}
#[test]
fn test_encodable() {
    let null_buffer = vec![0x05, 0x00];
    let null = Rc::new(DerNull::new());
    let asn1_encodable = null.to_asn1_encodable();
    {
        let buffer = asn1_encodable.get_encoded().expect("fail");
        assert_eq!(2, buffer.len());
        assert_eq!(null_buffer, buffer);
    }
    {
        let buffer = asn1_encodable.get_encoded_with_encoding(DER).expect("fail");
        assert_eq!(2, buffer.len());
        assert_eq!(null_buffer, buffer);
    }
    {
        let mut buffer = Vec::<u8>::new();
        let length = asn1_encodable.encode_to(&mut buffer).expect("fail");
        assert_eq!(2, length);
        assert_eq!(null_buffer, buffer);
    }
    {
        let mut buffer = Vec::<u8>::new();
        let length = asn1_encodable
            .encode_to_with_encoding(&mut buffer, DER)
            .expect("fail");
        assert_eq!(2, length);
        assert_eq!(null_buffer, buffer);
    }
}
