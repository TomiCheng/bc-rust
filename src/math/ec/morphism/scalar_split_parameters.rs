use crate::math::BigInteger;

pub struct ScalarSplitParameters {
    v1_a: BigInteger,
    v1_b: BigInteger,
    v2_a: BigInteger,
    v2_b: BigInteger,
    g1: BigInteger,
    g2: BigInteger,
    bits: usize,
}

impl ScalarSplitParameters {
    pub fn create(
        v1: [BigInteger; 2],
        v2: [BigInteger; 2],
        g1: BigInteger,
        g2: BigInteger,
        bits: usize,
    ) -> Self {
        let [v1_a, v1_b] = v1;
        let [v2_a, v2_b] = v2;
        ScalarSplitParameters {
            v1_a,
            v1_b,
            v2_a,
            v2_b,
            g1,
            g2,
            bits,
        }
    }
    pub fn v1_a(&self) -> &BigInteger {
        &self.v1_a
    }
    pub fn v1_b(&self) -> &BigInteger {
        &self.v1_b
    }
    pub fn v2_a(&self) -> &BigInteger {
        &self.v2_a
    }
    pub fn v2_b(&self) -> &BigInteger {
        &self.v2_b
    }
    pub fn g1(&self) -> &BigInteger {
        &self.g1
    }
    pub fn g2(&self) -> &BigInteger {
        &self.g2
    }
    pub fn bits(&self) -> usize {
        self.bits
    }
}
