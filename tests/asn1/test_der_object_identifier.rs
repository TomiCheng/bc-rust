use std::io;

use bc_rust::asn1;
use bc_rust::asn1::asn1_object;
use bc_rust::asn1::Asn1Encodable;
use bc_rust::util::encoders::hex;

#[test]
fn record_check_1() {
    let req = hex::to_decode_with_str("0603813403").unwrap();
    recode_check("2.100.3", &req);
}

#[test]
fn record_check_2() {
    let req = hex::to_decode_with_str("06082A36FFFFFFDD6311").unwrap();
    recode_check("1.2.54.34359733987.17", &req);
}

#[test]
fn record_check_value() {
    check_valid("0.1");
    check_valid("1.1.127.32512.8323072.2130706432.545460846592.139637976727552.35747322042253312.9151314442816847872");
    check_valid("1.2.123.12345678901.1.1.1");
    check_valid("2.25.196556539987194312349856245628873852187.1");
    check_valid("0.0");
    check_valid("0.0.1");
    check_valid("0.39");
    check_valid("0.39.1");
    check_valid("1.0");
    check_valid("1.0.1");
    check_valid("1.39");
    check_valid("1.39.1");
    check_valid("2.0");
    check_valid("2.0.1");
    check_valid("2.40");
    check_valid("2.40.1");
}

#[test]
fn check_invalid_1() {
    check_invalid("0");
    check_invalid("1");
    check_invalid("2");
    check_invalid("3.1");
    check_invalid("..1");
    check_invalid("192.168.1.1");
    check_invalid(".123452");
    check_invalid("1.");
    check_invalid("1.345.23.34..234");
    check_invalid("1.345.23.34.234.");
    check_invalid(".12.345.77.234");
    check_invalid(".12.345.77.234.");
    check_invalid("1.2.3.4.A.5");
    check_invalid("1,2");
    check_invalid("0.40");
    check_invalid("0.40.1");
    check_invalid("0.100");
    check_invalid("0.100.1");
    check_invalid("1.40");
    check_invalid("1.40.1");
    check_invalid("1.100");
    check_invalid("1.100.1");
}

#[test]
fn test_branch_check() {
    check_branch("1.1", "2.2");
}

#[test]
fn test_expected() {
    check_expected("1.1", "1.1", false);
    check_expected("1.1", "1.2", false);
	check_expected("1.1", "1.2.1", false);
	check_expected("1.1", "2.1", false);
	check_expected("1.1", "1.11", false);
	check_expected("1.12", "1.1.2", false);
	check_expected("1.1", "1.1.1", true);
	check_expected("1.1", "1.1.2", true);
    check_expected("1.2.3.4.5.6", "1.2.3.4.5.6", false);
    check_expected("1.2.3.4.5.6", "1.2.3.4.5.6.7", true);
    check_expected("1.2.3.4.5.6", "1.2.3.4.5.6.7.8", true);
}

fn recode_check(oid: &str, req: &Vec<u8>) {
    let o = asn1::DerObjectIdentifier::with_str(oid).unwrap();
    let mut cursor = io::Cursor::new(req);
    let enc_o = asn1_object::from_read(&mut cursor).unwrap();

    assert_eq!(o, *enc_o);

    let bytes = o.get_der_encoded().unwrap();
    assert_eq!(req, bytes.as_slice());
}

fn check_valid(oid: &str) {
    assert!(asn1::DerObjectIdentifier::try_from_id(oid).is_some());

    let o = asn1::DerObjectIdentifier::with_str(oid).unwrap();
    assert_eq!(oid, o.id());
}

fn check_invalid(oid: &str) {
    assert!(asn1::DerObjectIdentifier::try_from_id(oid).is_none());
    assert!(asn1::DerObjectIdentifier::with_str(oid).is_err());
}

fn check_branch(stem: &str, branch: &str) {
    let mut expected = stem.to_string();
    expected.push('.');
    expected.push_str(branch);

    let binding = asn1::DerObjectIdentifier::with_str(stem)
        .unwrap()
        .branch(branch)
        .unwrap();
    let actual = binding
        .id();
    assert_eq!(&expected, actual);
}

fn check_expected(stem: &str, test: &str, expected: bool) {
    let stem_obj = asn1::DerObjectIdentifier::with_str(stem).unwrap();
    let test_obj = asn1::DerObjectIdentifier::with_str(test).unwrap();
    let actual = test_obj.on(&stem_obj);
    assert_eq!(expected, actual);
}
