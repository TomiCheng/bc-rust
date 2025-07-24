use std::sync::Arc;
use crate::math::BigInteger;
use crate::math::ec::EcFieldElementRc;
use crate::math::ec::EcFieldElement;
use crate::math::field::FiniteField;

pub const COORD_AFFINE: u8 = 0;
pub const COORD_HOMOGENEOUS: u8 = 1;
pub const COORD_JACOBIAN: u8 = 2;
pub const COORD_JACOBIAN_CHUDNOVSKY: u8 = 3;
pub const COORD_JACOBIAN_MODIFIED: u8 = 4;
pub const COORD_LAMBDA_AFFINE: u8 = 5;
pub const COORD_LAMBDA_PROJECTIVE: u8 = 6;
pub const COORD_SKEWED: u8 = 7;

// use crate::math::ec::{AbstractEcLookupTable, EcLookupTable, EcPoint};
// use crate::math::field::FiniteField;
//

pub type EcCurveRc = Arc<dyn EcCurve>;
pub trait EcCurve {
    fn create_field_element_from_big_integer(&self, x: &BigInteger) -> EcFieldElementRc;
}

pub(crate) struct EcCurveImpl {
    field: Box<dyn FiniteField>

}

impl EcCurveImpl {
    pub(crate) fn new(field: Box<dyn FiniteField>) -> Self {
        EcCurveImpl { field }
    }
}
//
// // todo
//
// // EC Lookup Table
// struct DefaultLookupTable<'a> {
//     outer: &'a dyn EcCurve,
//     table: Vec<u8>,
//     size: usize,
// }
//
// impl<'a> DefaultLookupTable<'a> {
//     pub fn new(outer: &'a dyn EcCurve, table: Vec<u8>, size: usize) -> Self {
//         DefaultLookupTable { outer, table, size }
//     }
// }
// // impl EcLookupTable for DefaultLookupTable {
// //     type Point = ();
// //
// //     fn size(&self) -> usize {
// //         todo!()
// //     }
// //
// //     fn lookup(&self, index: usize) -> Option<&Self::Point> {
// //         todo!()
// //     }
// //
// //     fn lookup_var(&self, index: usize) -> Option<&Self::Point> {
// //         todo!()
// //     }
// // }
// //
// // impl AbstractEcLookupTable for DefaultLookupTable {
// // }