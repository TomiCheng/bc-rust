use bc_rust::crypto::Digest;
use bc_rust::util::encoders::hex::to_decode_with_str;
use bc_rust::util::Memoable;

pub struct TestDigest<'a, T: Digest + Memoable + Clone> {
    digest: T,
    input: &'a[&'a str],
    results: &'a[&'a str]
}

impl <'a, T: Digest + Memoable + Clone> TestDigest<'a, T> {
    pub fn new(digest: T, input: &'a [&'a str], results: &'a [&'a str]) -> TestDigest<'a, T> {
        TestDigest {
            digest,
            input,
            results,
        }
    }

    pub fn get_algorithm_name(&self) -> &str {
        self.digest.get_algorithm_name()
    }

    pub fn perform_test(&mut self) {
        let mut res_buf = vec![0u8; self.digest.get_digest_size()];

        for i in 0..self.input.len() - 1 {
            let msg = to_vec(self.input[i]);
            let expected = to_decode_with_str(self.results[i]).expect("invalid hex string");
            self.test_vector(i , &mut res_buf, &msg, &expected);
        }

        let last_v = to_vec(self.input[self.input.len() - 1]);
        let last_digest = to_decode_with_str(self.results[self.input.len() - 1]).expect("invalid hex string");
        self.test_vector(self.input.len() - 1, &mut res_buf, &last_v, &last_digest);

        // clone test
        self.digest.block_update(&last_v[0..last_v.len() / 2]);

        // clone the digest
        let mut d = self.digest.clone();

        self.digest.block_update(&last_v[(last_v.len() / 2)..]);
        self.digest.do_final(&mut res_buf);

        assert_eq!(res_buf, last_digest, "{}: failing clone vector test", self.get_algorithm_name());

        d.block_update(&last_v[(last_v.len() / 2)..]);
        d.do_final(&mut res_buf);
        assert_eq!(res_buf, last_digest, "{}: failing second clone vector tes", self.get_algorithm_name());

        // memo test
        self.digest.block_update(&last_v[0..last_v.len() / 2]);

        // copy the digest
        let copy1 = self.digest.copy();
        let mut copy2 = copy1.copy();

        self.digest.block_update(&last_v[(last_v.len() / 2)..]);
        self.digest.do_final(&mut res_buf);

        assert_eq!(res_buf, last_digest, "{}: failing memo vector test", self.get_algorithm_name());

        self.digest.reset(&copy1);

        self.digest.block_update(&last_v[(last_v.len() / 2)..]);
        self.digest.do_final(&mut res_buf);

        assert_eq!(res_buf, last_digest, "{}: failing memo reset vector test", self.get_algorithm_name());


        copy2.block_update(&last_v[(last_v.len() / 2)..]);
        copy2.do_final(&mut res_buf);

        assert_eq!(res_buf, last_digest, "{}: failing memo copy vector test", self.get_algorithm_name());
    }

    fn test_vector(&mut self, count: usize, res_buf: &mut [u8], input: &[u8], expected: &[u8]) {
        self.digest.block_update(input);
        self.digest.do_final(res_buf);
    
        assert_eq!(res_buf, expected, "{}: vector {} failed got", self.get_algorithm_name(), count);
    }
}

fn to_vec(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}

