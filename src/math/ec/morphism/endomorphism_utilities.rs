use crate::math::big_integer::ONE;
use crate::math::BigInteger;
use crate::math::ec::morphism::ScalarSplitParameters;

pub fn decompose_scalar(p: &ScalarSplitParameters, k: &BigInteger) -> [BigInteger; 2] {
    let bits = p.bits();
    let b1 = calculate_b(k, p.g1(), bits);
    let b2 = calculate_b(k, p.g2(), bits);

    let a = k.subtract(&b1.multiply(p.v1_b()).add(&b2.multiply(p.v2_a())));
    let b = b1.multiply(p.v1_b()).add(&b2.multiply(p.v2_b())).negate();
    [a, b]
}

fn calculate_b(k: &BigInteger, g: &BigInteger, t: usize) -> BigInteger {
    let negative = g.sign() < 0;
    let mut b = k.multiply(&g.abs());
    let extra = b.test_bit(t - 1);
    b = b.shift_right(t as isize);
    if extra {
        b = b.add(&*ONE);
    }
    if negative {
        b.negate()
    } else {
        b
    }
}