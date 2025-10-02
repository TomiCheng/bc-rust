#![cfg(test)]

use rand::RngCore;
use crate::crypto::Digest;
use crate::security::digest_utilities::do_final;
use crate::util::encoders::hex::decode_to_vec;
use crate::util::Memorable;

pub(crate) fn test_digest<T: Digest +  Memorable>(mut digest: T, messages: &[&str], results: &[&str]) {
    let mut res_buf = vec![0u8; digest.digest_size()];
    for i in 0..messages.len() - 1 {
        let msg = messages[i].as_bytes().to_vec();
        let expected = decode_to_vec(results[i]).unwrap();
        test_vector(&mut digest, i, &mut res_buf, &msg, &expected);
    }

    let last_v = messages[messages.len() - 1].as_bytes().to_vec();
    let last_digest = decode_to_vec(results[messages.len() - 1]).unwrap();
    test_vector(&mut digest, messages.len() - 1, &mut res_buf, &last_v, &last_digest);

    // clone test
    digest.block_update(&last_v[0..last_v.len() / 2]).unwrap();

    // clone the digest
    let mut d = digest.clone();

    digest.block_update(&last_v[(last_v.len() / 2)..]).unwrap();
    digest.do_final(&mut res_buf).unwrap();

    assert_eq!(res_buf, last_digest, "{}: failing clone vector test", digest.algorithm_name());

    d.block_update(&last_v[(last_v.len() / 2)..]).unwrap();
    d.do_final(&mut res_buf).unwrap();
    assert_eq!(res_buf, last_digest, "{}: failing second clone vector tes", digest.algorithm_name());

    // memo test
    digest.block_update(&last_v[0..last_v.len() / 2]).unwrap();

    // copy the digest
    let copy1 = digest.clone();
    let mut copy2 = copy1.clone();

    digest.block_update(&last_v[(last_v.len() / 2)..]).unwrap();
    digest.do_final(&mut res_buf).unwrap();

    assert_eq!(res_buf, last_digest, "{}: failing memo vector test", digest.algorithm_name());

    digest.restore(&copy1).unwrap();
    digest.block_update(&last_v[(last_v.len() / 2)..]).unwrap();
    digest.do_final(&mut res_buf).unwrap();

    assert_eq!(res_buf, last_digest, "{}: failing memo reset vector test", digest.algorithm_name());

    copy2.block_update(&last_v[(last_v.len() / 2)..]).unwrap();
    copy2.do_final(&mut res_buf).unwrap();

    assert_eq!(res_buf, last_digest, "{}: failing memo copy vector test", digest.algorithm_name());

    span_consistency_tests(&mut digest);
}
fn test_vector<TDigest: Digest>(
    digest: &mut TDigest,
    count: usize,
    res_buf: &mut [u8],
    input: &[u8],
    expected: &[u8],
) {
    digest.block_update(input).unwrap();
    digest.do_final(res_buf).unwrap();

    assert_eq!(
        res_buf,
        expected,
        "{}: vector {} failed got",
        digest.algorithm_name(),
        count
    );
}
fn span_consistency_tests<TDigest: Digest>(digest: &mut TDigest) {
    let mut data = vec![0u8; 16 + 256];
    let mut random = rand::rng();
    random.fill_bytes(&mut data);
    for len in 0..=256 {
        let off: usize = rand::random_range(0..16);
        span_consistency_test(digest, &data[off..(off + len)]);
    }
}
fn span_consistency_test<TDigest: Digest>(digest: &mut TDigest, data: &[u8]) {
    digest.reset();
    let span_result1 = do_final(digest, data).unwrap();

    let mut pos = 0;
    while pos < data.len() {
        let next = 1 + rand::random_range(0..(data.len() - pos));
        digest.block_update(&data[pos..(pos + next)]).unwrap();
        pos += next;
    }

    let mut span_result2 = vec![0u8; digest.digest_size()];
    digest.do_final(&mut span_result2).unwrap();
    assert_eq!(span_result1, span_result2, "{}: span consistency test failed", digest.algorithm_name());
}
