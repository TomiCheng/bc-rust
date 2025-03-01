use std::rc::Rc;

use bc_rust::asn1::{Asn1Convertiable, DerBitString};

#[test]
fn test_zero_length_strings() {
    let s1 = Rc::new(DerBitString::with_pad_bits(&vec![], 0).expect("fail"));
    s1.get_bytes();

    let asn1_encodable = s1.to_asn1_encodable();
    let buffer = asn1_encodable.get_encoded().expect("fail");

    assert_eq!(buffer, vec![0x03, 0x01, 0x00]);

    let s2 = Rc::new(DerBitString::with_named_bits(0));
    let asn1_encodable = s2.to_asn1_encodable();
    let buffer2 = asn1_encodable.get_encoded().expect("fail");

    assert_eq!(buffer2, buffer);
}

#[test]
fn test_with_pad_bits_fail() {
    assert!(DerBitString::with_pad_bits(&vec![], 1).is_err());
    assert!(DerBitString::with_pad_bits(&vec![0], 8).is_err());
}