use crate::math::BigInteger;
use crate::math::ec::ec_curve::COORD_JACOBIAN_MODIFIED;
//use crate::math::ec::fp_field_element::{FpFieldElement, calculate_residue};
//use crate::math::ec::fp_point::FpPoint;
use crate::{BcError, Result};
use std::sync::Arc;

/// Elliptic curve over Fp
pub struct FpCurve {
    q: BigInteger,
    r: Option<BigInteger>,
    //infinity: Arc<FpPoint>,

}

impl FpCurve {
    const FP_DEFAULT_COORDS: u8 = COORD_JACOBIAN_MODIFIED;
    //     pub fn new(
    //         q: BigInteger,
    //         a: BigInteger,
    //         b: BigInteger,
    //         order: BigInteger,
    //         cofactor: BigInteger,
    //     ) -> Result<Arc<Self>> {
    //         let r = calculate_residue(&q)?;
    //         let a = from_big_integer(&q, r.as_ref(), a)?;
    //         let b = from_big_integer(&q, r.as_ref(), b)?;
    //         let curve = Arc::new_cyclic(|weak_parent| FpCurve {
    //             q,
    //             r,
    //             infinity: FpPoint::with_curve(weak_parent.clone()),
    //             a,
    //             b,
    //             order,
    //             cofactor,
    //             coord: Self::FP_DEFAULT_COORDS,
    //         });
    //         Ok(curve)
    //     }
    //
    //     pub fn create_point(self: &Arc<Self>, x: BigInteger, y: BigInteger) -> Result<Arc<FpPoint>> {
    //         let x = self.create_field_element(x)?;
    //         let y = self.create_field_element(y)?;
    //         Ok(FpPoint::with_curve_x_y(Arc::downgrade(&self), Some(x), Some(y))?)
    //     }
    //     pub fn create_field_element(&self, x: BigInteger) -> Result<FpFieldElement> {
    //         let field_element = from_big_integer(&self.q, self.r.as_ref(), x)?;
    //         Ok(field_element)
    //     }
    //     pub fn infinity(&self) -> Arc<FpPoint> {
    //         self.infinity.clone()
    //     }


}
//
// fn from_big_integer(
//     q: &BigInteger,
//     r: Option<&BigInteger>,
//     x: BigInteger,
// ) -> Result<FpFieldElement> {
//     if x.sign() < 0 || &x >= q {
//         return Err(BcError::with_invalid_argument(
//             "value invalid for Fp field element",
//         ));
//     }
//     Ok(FpFieldElement::new(q.clone(), r.cloned(), x))
// }
//
// // TODO
