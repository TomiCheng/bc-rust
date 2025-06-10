
use std::random::RandomSource;
use crate::crypto::digests::Digest;
use crate::util::CloneableState;
use crate::util::encoders::hex::to_decode_with_str;
use crate::security::digest_utilities::do_final;

pub(crate) fn test_digest<T: Digest + CloneableState + Clone>(mut digest: T, messages: &[&str], results: &[&str]) {
    let mut res_buf = vec![0u8; digest.get_digest_size()];
    for i in 0..messages.len() - 1 {
         let msg = to_vec(messages[i]);
         let expected = to_decode_with_str(results[i]).expect("invalid hex string");
         test_vector(&mut digest, i , &mut res_buf, &msg, &expected);
    }
    
    let last_v = to_vec(messages[messages.len() - 1]);
    let last_digest = to_decode_with_str(results[messages.len() - 1]).expect("invalid hex string");
    test_vector(&mut digest, messages.len() - 1, &mut res_buf, &last_v, &last_digest);
    
    // clone test
    digest.block_update(&last_v[0..last_v.len() / 2]);
    
    // clone the digest
    let mut d = digest.clone();
    
    digest.block_update(&last_v[(last_v.len() / 2)..]);
    digest.do_final(&mut res_buf);
    
    assert_eq!(res_buf, last_digest, "{}: failing clone vector test", digest.algorithm_name());
    
    d.block_update(&last_v[(last_v.len() / 2)..]);
    d.do_final(&mut res_buf);
    assert_eq!(res_buf, last_digest, "{}: failing second clone vector tes", digest.algorithm_name());
    
    // memo test
    digest.block_update(&last_v[0..last_v.len() / 2]);
    
    // copy the digest
    let copy1 = digest.copy();
    let mut copy2 = copy1.copy();
    
    digest.block_update(&last_v[(last_v.len() / 2)..]);
    digest.do_final(&mut res_buf);
    
    assert_eq!(res_buf, last_digest, "{}: failing memo vector test", digest.algorithm_name());
    
    digest.restore(&copy1);
    digest.block_update(&last_v[(last_v.len() / 2)..]);
    digest.do_final(&mut res_buf);
    
    assert_eq!(res_buf, last_digest, "{}: failing memo reset vector test", digest.algorithm_name());
    
    
    copy2.block_update(&last_v[(last_v.len() / 2)..]);
    copy2.do_final(&mut res_buf);
    
    assert_eq!(res_buf, last_digest, "{}: failing memo copy vector test", digest.algorithm_name());
    
    span_consistency_tests(&mut digest);
}

fn to_vec(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}
fn test_vector<TDigest: Digest>(digest: &mut TDigest, count: usize, res_buf: &mut [u8], input: &[u8], expected: &[u8]) {
    digest.block_update(input);
    digest.do_final(res_buf);

    assert_eq!(res_buf, expected, "{}: vector {} failed got", digest.algorithm_name(), count);
}
fn span_consistency_tests<TDigest: Digest>(digest: &mut TDigest) {
    let mut data = vec![0u8; 16 + 256];
    let mut random = std::random::DefaultRandomSource::default();
    random.fill_bytes(&mut data);
    for len in 0..=256 {
        let off: usize = std::random::random::<usize>() % 16;
        span_consistency_test(digest,&data[off..(off + len)]);
    }
}
fn span_consistency_test<TDigest: Digest>(digest: &mut TDigest, data: &[u8]) {
    digest.reset();
    let span_result1 = do_final(digest, data);

    let mut pos = 0;
    while pos < data.len() {
        let next = 1 + std::random::random::<usize>() % (data.len() - pos);
        digest.block_update(&data[pos..(pos + next)]);
        pos += next;
    }

    let mut span_result2 = vec![0u8; digest.get_digest_size()];
    digest.do_final(&mut span_result2);
    assert_eq!(span_result1, span_result2, "{}: span consistency test failed", digest.algorithm_name());
}