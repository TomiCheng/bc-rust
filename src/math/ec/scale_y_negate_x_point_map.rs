use crate::math::ec::{EcFieldElementRc, EcPointMap};

pub struct ScaleYNegateXPointMap {
    scale: EcFieldElementRc,
}
impl ScaleYNegateXPointMap {
    pub fn new(scale: EcFieldElementRc) -> Self {
        ScaleYNegateXPointMap { scale }
    }
}

impl EcPointMap for ScaleYNegateXPointMap {
    fn map(&self) {
        todo!()
    }
}