//! Standard vector test for SHA-1 from "Handbook of Applied Cryptography", page 345.

use bc_rust::crypto::digests::Sha256Digest;

use super::TestDigest;

const MESSAGES: [&str; 4] = [
    "",
    "a",
    "abc",
    "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
];

const DIGESTS: [&str; 4] = [
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb",
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
];

#[test]
fn test_functions() {
    let mut test = TestDigest::new(Sha256Digest::new(), &MESSAGES, &DIGESTS);
    test.perform_test();
}
