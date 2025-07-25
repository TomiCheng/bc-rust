use crate::math::big_integer::ONE;
use crate::math::BigInteger;
use crate::{Result};
use crate::util::big_integers;

#[derive(Debug, Clone)]
pub struct FpFieldElement {
    q: BigInteger,
    r: Option<BigInteger>,
    x: BigInteger,
}
impl FpFieldElement {
    pub(crate) fn new(q: BigInteger, r: Option<BigInteger>, x: BigInteger) -> Self {
        FpFieldElement { q, r, x }
    }

    pub fn to_big_integer(&self) -> BigInteger {
        self.x.clone()
    }
    pub fn is_one(&self) -> bool {
        self.bit_length() == 1
    }
    pub fn is_zero(&self) -> bool {
        self.x.sign() == 0
    }
    pub fn bit_length(&self) -> usize {
        self.x.bit_length()
    }
    pub fn multiply(&self, b: &Self) -> Result<Self> {
        Ok(Self::new(self.q.clone(), self.r.clone(), self.mod_multiply(&self.x, &b.x)?))
    }
    pub fn invert(&self) -> Result<Self> {
        Ok(Self::new(self.q.clone(), self.r.clone(), self.mod_inverse(&self.x)?))
    }
    pub fn square(&self) -> Result<Self> {
        Ok(Self::new(self.q.clone(), self.r.clone(), self.mod_multiply(&self.x, &self.x)?))
    }
    pub fn subtract(&self, b: &Self) -> Self {
        Self::new(self.q.clone(), self.r.clone(), self.mod_subtract(&self.x, &b.x))
    }
    pub fn divide(&self, b: &Self) -> Result<Self> {
        Ok(Self::new(self.q.clone(), self.r.clone(), self.mod_multiply(&self.x, &self.mod_inverse(&b.x)?)?))
    }
    pub fn add(&self, b: &Self) -> Self {
        Self::new(self.q.clone(), self.r.clone(), self.mod_add(&self.x, &b.x))
    }
    pub fn multiply_minus_product(&self, b: &Self, x: &Self, y: &Self) -> Result<Self> {
        let ax = &self.x;
        let bx = &b.x;
        let xx = &x.x;
        let yx = &y.x;
        let ab = ax.multiply(bx);
        let xy = xx.multiply(yx);
        Ok(Self::new(self.q.clone(), self.r.clone(), self.mod_reduce(&ab.subtract(&xy))?))
    }
    pub fn negate(&self) -> Self {
        if self.x.sign() == 0 {
            self.clone()
        } else {
            Self::new(self.q.clone(), self.r.clone(), self.q.subtract(&self.x))
        }
    }
    fn mod_multiply(&self, x1: &BigInteger, x2: &BigInteger) -> Result<BigInteger> {
        self.mod_reduce(&x1.multiply(x2))
    }
    fn mod_reduce(&self, x: &BigInteger) -> Result<BigInteger> {
        let mut x = x.clone();
        if let Some(r) = &self.r {
            let negative = x.sign() < 0;
            if negative {
                x = x.abs();
            }
            let q_len = self.q.bit_length();
            if r.sign() > 0 {
                let q_mod = (*ONE).shift_left(q_len as isize);
                let r_is_one = r == &(*ONE);

                while x.bit_length() > (q_len + 1) {
                    let mut u = x.shift_right(q_len as isize);
                    let v = x.remainder(&q_mod)?;
                    if r_is_one {
                        u = u.multiply(&r);
                    }
                    x = u.add(&v);
                }
            } else {
                let d = ((q_len - 1) & 31) + 1;
                let mu = r.negate();
                let u = mu.multiply(&x.shift_right((q_len - d) as isize));
                let quot = u.shift_right((q_len + d) as isize);
                let mut v = quot.multiply(&self.q);
                let bk1 = (*ONE).shift_left((q_len + d) as isize);
                v = v.remainder(&bk1)?;
                x = x.remainder(&bk1)?;
                x = x.subtract(&v);
                if x.sign() < 0 {
                    x = x.add(&bk1);
                }
            }
            while x >= self.q {
                x = x.subtract(&self.q);
            }
            if negative && x.sign() != 0 {
                x = self.q.subtract(&x);
            }
        } else {
            x = x.modulus(&self.q)?;
        }
        Ok(x)
    }
    fn mod_inverse(&self, x: &BigInteger) -> Result<BigInteger> {
        big_integers::mod_odd_inverse(&self.q, x)
    }
    fn mod_subtract(&self, x1: &BigInteger, x2: &BigInteger) -> BigInteger {
        let mut x3 = x1.subtract(x2);
        if x3.sign() < 0 {
            x3 = x3.add(&self.q)
        }
        x3
    }
    fn mod_add(&self, x1: &BigInteger, x2: &BigInteger) -> BigInteger {
        let mut x3 = x1.add(x2);
        if x3 >= self.q {
            x3 = x3.subtract(&self.q)
        }
        x3
    }
}
impl PartialEq for FpFieldElement {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x
    }
}

//     fn mod_double(&self, x: &BigInteger) -> BigInteger {
//         let mut i2x = x.shift_left(1);
//         if i2x >= self.q {
//             i2x = i2x.subtract(&self.q);
//         }
//         i2x
//     }
//     fn mod_half_abs(&self, x: &BigInteger) -> BigInteger {
//         let mut x = x.clone();
//         if x.test_bit(0) {
//             x = self.q.subtract(&x);
//         }
//         x.shift_right(1)
//     }




//     fn check_sqrt(&self, z: Self) -> Result<Option<Self>> {
//         if z.square()? == *self {
//             Ok(Some(z))
//         } else {
//             Ok(None)
//         }
//     }
//     fn lucas_sequence(&self, p: &BigInteger, q: &BigInteger, k: &BigInteger) -> Result<(BigInteger, BigInteger)> {
//         let n = k.bit_length();
//         let s = k.get_lowest_set_bit() as usize;
// 
//         debug_assert!(k.test_bit(s));
// 
//         let mut uh = (*ONE).clone();
//         let mut vl = (*TWO).clone();
//         let mut vh = p.clone();
//         let mut ql = (*ONE).clone();
//         let mut qh = (*ONE).clone();
// 
//         for j in ((n - 1)..(s + 1)).rev() {
//             ql = self.mod_multiply(&ql, &qh)?;
// 
//             if k.test_bit(j) {
//                 qh = self.mod_multiply(&ql, q)?;
//                 uh = self.mod_multiply(&uh, &vh)?;
//                 vl = self.mod_reduce(&vh.multiply(&vl).subtract(&p.multiply(&ql)))?;
//                 vh = self.mod_reduce(&vh.multiply(&vh).subtract(&qh.shift_left(1)))?;
//             } else {
//                 qh = ql.clone();
//                 uh = self.mod_reduce(&uh.multiply(&vl).subtract(&ql))?;
//                 vh = self.mod_reduce(&uh.multiply(&vl).subtract(&p.multiply(&ql)))?;
//                 vl = self.mod_reduce(&vl.multiply(&vl).subtract(&ql.shift_left(1)))?;
//             }
//         }
// 
//         ql = self.mod_multiply(&ql, &qh)?;
//         qh = self.mod_multiply(&ql, q)?;
//         uh = self.mod_reduce(&uh.multiply(&vl).subtract(&ql))?;
//         vl = self.mod_reduce(&vh.multiply(&vl).subtract(&p.multiply(&ql)))?;
//         ql = self.mod_multiply(&ql, &qh)?;
// 
//         for _ in 1..s {
//             uh = self.mod_multiply(&uh, &vl)?;
//             vl = self.mod_reduce(&vl.multiply(&vl).subtract(&ql.shift_left(1)))?;
//             ql = self.mod_multiply(&ql, &ql)?;
//         }
//         Ok((uh, vl))
//     }
//}
// 
// impl EcFieldElement for FpFieldElement {
//     fn big_integer(&self) -> &BigInteger {
//         &self.x
//     }
//     /// return the field name for this field.
//     fn field_name(&self) -> String {
//         "Fp".to_string()
//     }
// 
//     fn field_size(&self) -> usize {
//         self.q.bit_length()
//     }
// 

// 
//     fn add_one(&self) -> Self {
//         let mut x2 = self.x.add(&(*ONE));
//         if x2 == self.q {
//             x2 = (*ZERO).clone();
//         }
//         Self::new(self.q.clone(), self.r.clone(), x2)
//     }
// 

// 

// 

// 

// 

// 

// 
//     fn sqrt(&self) -> Result<Option<Self>>
//     where
//         Self: Sized
//     {
//         if self.is_zero() || self.is_one() {
//             return Ok(Some(self.clone()));
//         }
// 
//         if !self.q.test_bit(0) {
//             return Err(BcError::with_invalid_operation("even value of q"));
//         }
// 
//         // q == 4m + 3
//         if self.q.test_bit(1) {
//             let e = self.q.shift_right(2).add(&(*ONE));
//             return self.check_sqrt(Self::new(self.q.clone(), self.r.clone(), self.x.mod_pow(&e, &self.q)?))
//         }
// 
//         // q == 8m + 5
//         if self.q.test_bit(2) {
//             let t1 = self.x.mod_pow(&self.q.shift_right(3), &self.q)?;
//             let t2 = self.mod_multiply(&t1, &self.x)?;
//             let t3 = self.mod_multiply(&t2, &t1)?;
// 
//             if t3 == *ONE {
//                 return self.check_sqrt(Self::new(self.q.clone(), self.r.clone(), t2));
//             }
// 
//             let t4 = (*TWO).mod_pow(&self.q.shift_right(2), &self.q)?;
//             let y = self.mod_multiply(&t2, &t4)?;
// 
//             return self.check_sqrt(Self::new(self.q.clone(), self.r.clone(), y));
//         }
// 
//         // q == 8m + 1
//         let legendre_exponent = self.q.shift_right(1);
//         if !(self.x.mod_pow(&legendre_exponent, &self.q)? == *ONE) {
//             return Ok(None);
//         }
// 
//         let x = &self.x;
//         let four_x = self.mod_double(&self.mod_double(&x));
//         let k = legendre_exponent.add(&(*ONE));
//         let q_minus_one = self.q.subtract(&(*ONE));
//         let mut u: BigInteger;
//         let mut v: BigInteger;
//         loop {
//             let mut p;
//             loop {
//                 p = BigInteger::with_arbitrary(self.q.bit_length());
//                 if !(p >= self.q || self.mod_reduce(&p.multiply(&p).subtract(&four_x).mod_pow(&legendre_exponent, &self.q)?)? != q_minus_one) {
//                     break;
//                 }
//             }
// 
//             (u, v) = self.lucas_sequence(&p, &self.x, &k)?;
//             if self.mod_multiply(&v, &v)? == four_x {
//                 return Ok(Some(Self::new(self.q.clone(), self.r.clone(), self.mod_half_abs(&v))));
//             }
// 
//             if !(u == (*ONE) || u == q_minus_one) {
//                 break;
//             }
//         }
//         Ok(None)
//     }
// 

// 
//     fn multiply_plus_product(&self, b: &Self, x: &Self, y: &Self) -> Result<Self>
//     where
//         Self: Sized
//     {
//         let ax = &self.x;
//         let bx = b.big_integer();
//         let xx = x.big_integer();
//         let yx = y.big_integer();
//         let ab = ax.multiply(bx);
//         let xy = xx.multiply(yx);
//         let mut sum = ab.add(&xy);
//         if let Some(r) = &self.r && r.sign() < 0 && sum.bit_length() > (self.q.bit_length() << 1) {
//             sum = sum.subtract(&self.q.shift_left(self.q.bit_length() as isize));
//         }
//         Ok(Self::new(self.q.clone(), self.r.clone(), self.mod_reduce(&sum)?))
//     }
// 
//     fn square_minus_product(&self, x: &Self, y: &Self) -> Result<Self>
//     where
//         Self: Sized
//     {
//         let ax = &self.x;
//         let xx = x.big_integer();
//         let yx = y.big_integer();
//         let aa = ax.multiply(ax);
//         let xy = xx.multiply(yx);
//         Ok(Self::new(self.q.clone(), self.r.clone(), self.mod_reduce(&aa.subtract(&xy))?))
//     }
// 
//     fn square_plus_product(&self, x: &Self, y: &Self) -> Result<Self>
//     where
//         Self: Sized
//     {
//         let ax = &self.x;
//         let xx = x.big_integer();
//         let yx = y.big_integer();
//         let aa = ax.multiply(ax);
//         let xy = xx.multiply(yx);
//         let mut sum = aa.add(&xy);
//         if let Some(r) = &self.r && r.sign() < 0 && sum.bit_length() > (self.q.bit_length() << 1) {
//             sum = sum.subtract(&self.q.shift_left(self.q.bit_length() as isize));
//         }
//         Ok(Self::new(self.q.clone(), self.r.clone(), self.mod_reduce(&sum)?))
//     }
// }
// impl Display for FpFieldElement {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{}", self.big_integer().to_string_radix(16).unwrap())
//     }
// }
// impl PartialEq for FpFieldElement {
//     fn eq(&self, other: &Self) -> bool {
//         self.q == other.q && self.x == other.x
//     }
// }
// impl Hash for FpFieldElement {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         self.q.hash(state);
//         self.x.hash(state);
//     }
// }
// 
pub(crate) fn calculate_residue(p: &BigInteger) -> Result<Option<BigInteger>> {
    let bit_length = p.bit_length();
    if bit_length >= 96 {
        let first_word = p.shift_right(bit_length as isize - 64);
        if first_word.as_i64() == -1 {
            return Ok(Some((*ONE).shift_left(bit_length as isize).subtract(p)))
        }
        if bit_length & 7 == 0 {
            return Ok(Some((*ONE).shift_left((bit_length << 1) as isize).divide(p)?.negate()))
        }
    }
    Ok(None)
}