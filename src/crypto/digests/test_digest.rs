
use rand::RngCore;
use crate::crypto::Digest;
use crate::util::Memoable;
use crate::util::encoders::hex::to_decode_with_str;
use crate::security::digest_utilities::do_final;

pub(crate) fn test_digest<T: Digest + Memoable>(mut digest: T, messages: &[&str], results: &[&str]) {
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
    digest.block_update(&last_v[0..last_v.len() / 2]).unwrap();
    
    // clone the digest
    let mut d = digest.copy();
    
    digest.block_update(&last_v[(last_v.len() / 2)..]).unwrap();
    digest.do_final(&mut res_buf).unwrap();
    
    assert_eq!(res_buf, last_digest, "{}: failing clone vector test", digest.algorithm_name());
    
    d.block_update(&last_v[(last_v.len() / 2)..]).unwrap();
    d.do_final(&mut res_buf).unwrap();
    assert_eq!(res_buf, last_digest, "{}: failing second clone vector tes", digest.algorithm_name());
    
    // memo test
    digest.block_update(&last_v[0..last_v.len() / 2]).unwrap();
    
    // copy the digest
    let copy1 = digest.copy();
    let mut copy2 = copy1.copy();
    
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
pub(crate) fn test_digest_reset<T: Digest>(mut digest: T) -> bool {
    const DATA_LEN: usize = 100;
    
    // obtain some random data
    let mut my_data = vec![0u8; DATA_LEN];
    let mut random_source = rand::rng();
    random_source.fill_bytes(&mut my_data);
    
    // update and finalise digest
    let my_hash_len = digest.get_digest_size();
    let mut my_first = vec![0u8; my_hash_len];
    digest.block_update(&my_data).unwrap();
    digest.do_final(&mut my_first).unwrap();
    
    // reuse the digest
    let mut my_second = vec![0u8; my_hash_len];
    digest.block_update(&my_data).unwrap();
    digest.do_final(&mut my_second).unwrap();
    
    my_first.eq(&my_second)
}
fn to_vec(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}
fn test_vector<TDigest: Digest>(digest: &mut TDigest, count: usize, res_buf: &mut [u8], input: &[u8], expected: &[u8]) {
    digest.block_update(input).unwrap();
    digest.do_final(res_buf).unwrap();

    assert_eq!(res_buf, expected, "{}: vector {} failed got", digest.algorithm_name(), count);
}
fn span_consistency_tests<TDigest: Digest>(digest: &mut TDigest) {
    let mut data = vec![0u8; 16 + 256];
    let mut random = rand::rng();
    random.fill_bytes(&mut data);
    for len in 0..=256 {
        let off: usize = rand::random_range(0..16); 
        span_consistency_test(digest,&data[off..(off + len)]);
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

    let mut span_result2 = vec![0u8; digest.get_digest_size()];
    digest.do_final(&mut span_result2).unwrap();
    assert_eq!(span_result1, span_result2, "{}: span consistency test failed", digest.algorithm_name());
}