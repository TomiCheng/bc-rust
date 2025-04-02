use bc_rust::asn1::{self, Asn1BitString};
use bc_rust::asn1::Asn1Encodable;
use bc_rust::util::encoders::hex;
use bc_rust::asn1::asn1_object;

#[test]
fn test_zero_length_strings() {
    let s1 = asn1::Asn1BitString::with_pad_bits(&vec![], 0).expect("fail");
    s1.get_bytes();

    let buffer = s1.get_encoded().expect("fail");
    assert_eq!(buffer, vec![0x03, 0x01, 0x00]);

    let s2 = asn1::Asn1BitString::with_named_bits(0);
    let buffer2 = s2.get_encoded().expect("fail");
    assert_eq!(buffer2, buffer);
}

#[test]
fn test_with_pad_bits_fail() {
    assert!(asn1::Asn1BitString::with_pad_bits(&vec![], 1).is_err());
    assert!(asn1::Asn1BitString::with_pad_bits(&vec![0], 8).is_err());
}

#[test]
fn test_random_pad_bits() {
    let test = hex::to_decode_with_str("030206c0").unwrap();
    let test1 = hex::to_decode_with_str("030206f0").unwrap();
    let test2 = hex::to_decode_with_str("030206c1").unwrap();
    let test3 = hex::to_decode_with_str("030206c7").unwrap();
    let test4 = hex::to_decode_with_str("030206d1").unwrap();
    check_encoding(&test, &test1);
    check_encoding(&test, &test2);
    check_encoding(&test, &test3);
    check_encoding(&test, &test4);
}

fn check_encoding(der_data: &Vec<u8>, dl_data: &Vec<u8>) {
    let dl_asn1_object = asn1_object::from_read(&mut dl_data.as_slice()).unwrap();
    let dl_buffer = dl_asn1_object.get_encoded().unwrap();
    assert_ne!(der_data, &dl_buffer);

    let dl = Asn1BitString::get_instance(dl_data).unwrap();

    //dl.
}