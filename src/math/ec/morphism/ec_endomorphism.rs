use std::sync::Arc;
use crate::math::ec::EcPointMapRc;

pub type EcEndomorphismRc = Arc<dyn EcEndomorphism>;
pub trait EcEndomorphism {
    fn point_map(&self) -> EcPointMapRc;
    fn has_efficient_point_map(&self) -> bool;
}