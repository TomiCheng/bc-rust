use bc_rust::asn1;
use bc_rust::asn1::Asn1Encodable;

#[test]
fn test_zero_length_strings() {
    let s1 = asn1::DerBitString::with_pad_bits(&vec![], 0).expect("fail");
    s1.get_bytes();

    let buffer = s1.get_encoded().expect("fail");
    assert_eq!(buffer, vec![0x03, 0x01, 0x00]);

    let s2 = asn1::DerBitString::with_named_bits(0);
    let buffer2 = s2.get_encoded().expect("fail");
    assert_eq!(buffer2, buffer);
}

#[test]
fn test_with_pad_bits_fail() {
    assert!(asn1::DerBitString::with_pad_bits(&vec![], 1).is_err());
    assert!(asn1::DerBitString::with_pad_bits(&vec![0], 8).is_err());
}
