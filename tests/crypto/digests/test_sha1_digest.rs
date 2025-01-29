//! Standard vector test for SHA-1 from "Handbook of Applied Cryptography", page 345.

use bc_rust::crypto::digests::Sha1Digest;

use super::TestDigest;

const MESSAGES: [&str; 4] = ["", "a", "abc", "abcdefghijklmnopqrstuvwxyz"];

const DIGESTS: [&str; 4] = [
    "da39a3ee5e6b4b0d3255bfef95601890afd80709",
    "86f7e437faa5a7fce15d1ddcb9eaeaea377667b8",
    "a9993e364706816aba3e25717850c26c9cd0d89d",
    "32d10c7b8cf96570ca04ce37f2a19d84240d3a89",
];

#[test]
fn test_functions() {
    let mut test = TestDigest::new(Sha1Digest::new(), &MESSAGES, &DIGESTS);
    test.perform_test();
}
