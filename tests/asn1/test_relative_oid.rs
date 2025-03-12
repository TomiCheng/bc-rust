use std::io;

use bc_rust::asn1;
use bc_rust::asn1::Asn1Encodable;
use bc_rust::util::encoders::hex;

#[test]
fn record_check_1() {
    let req = hex::to_decode_with_str("0D03813403").unwrap();
    recode_check("180.3", &req);
}
#[test]
fn record_check_2() {
    let req = hex::to_decode_with_str("0D082A36FFFFFFDD6311").unwrap();
    recode_check("42.54.34359733987.17", &req);
}

#[test]
fn test_check_valid() {
    check_valid("0");
    check_valid("37");
    check_valid("0.1");
    check_valid("1.0");
    check_valid("1.0.2");
    check_valid("1.0.20");
    check_valid("1.0.200");
    check_valid("1.1.127.32512.8323072.2130706432.545460846592.139637976727552.35747322042253312.9151314442816847872");
    check_valid("1.2.123.12345678901.1.1.1");
    check_valid("2.25.196556539987194312349856245628873852187.1");
    check_valid("3.1");
    check_valid("37.196556539987194312349856245628873852187.100");
    check_valid("192.168.1.1");
}

#[test]
fn test_check_invalid() {
    check_invalid("00");
    check_invalid("0.01");
    check_invalid("00.1");
    check_invalid("1.00.2");
    check_invalid("1.0.02");
    check_invalid("1.2.00");
    check_invalid(".1");
    check_invalid("..1");
    check_invalid("3..1");
    check_invalid(".123452");
    check_invalid("1.");
    check_invalid("1.345.23.34..234");
    check_invalid("1.345.23.34.234.");
    check_invalid(".12.345.77.234");
    check_invalid(".12.345.77.234.");
    check_invalid("1.2.3.4.A.5");
    check_invalid("1,2");
}

#[test]
fn test_branch_check() {
    branch_check("1.1", "2.2");
}

fn recode_check(oid: &str, req: &Vec<u8>) {
    let o: asn1::Asn1Object = asn1::Asn1RelativeOidImpl::with_str(oid).unwrap().into();
    let mut cursor = io::Cursor::new(req);
    let enc_o = asn1::Asn1Object::parse(&mut cursor).unwrap();

    assert_eq!(o, enc_o);

    let bytes = o.get_der_encoded().unwrap();
    assert_eq!(req, bytes.as_slice());
}

fn check_valid(oid: &str) {
    assert!(asn1::Asn1RelativeOidImpl::try_from_id(oid).is_some());
}

fn check_invalid(oid: &str) {
    assert!(asn1::Asn1RelativeOidImpl::try_from_id(oid).is_none());
    assert!(asn1::Asn1RelativeOidImpl::with_str(oid).is_err());
}

fn branch_check(stem: &str, branch: &str) {
    let expected = format!("{}.{}", stem, branch);
    let actual = asn1::Asn1RelativeOidImpl::with_str(stem)
        .unwrap()
        .branch(branch)
        .unwrap()
        .get_id();
    assert_eq!(expected, actual);
}
