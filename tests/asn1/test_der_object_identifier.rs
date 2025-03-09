use bc_rust::{asn1::{Asn1Object, DerObjectIdentifierImpl}, util::encoders::hex::to_decode_with_str};

#[test]
fn record_check_1() {
    let req = to_decode_with_str("0603813403").unwrap();
    recode_check("2.100.3", &req);
}

fn recode_check(oid: &str, mut req: &[u8]) {
    let o: Asn1Object = DerObjectIdentifierImpl::with_str(oid).unwrap().into();
    let enc_o = Asn1Object::parse(&mut req).unwrap();
}
