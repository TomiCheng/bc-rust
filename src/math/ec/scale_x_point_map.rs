use crate::math::ec::{EcFieldElementRc, EcPointMap};

pub struct ScaleXPointMap {
    scale: EcFieldElementRc,
}

impl ScaleXPointMap {
    pub fn new(scale: EcFieldElementRc) -> Self {
        ScaleXPointMap { scale }
    }
}

impl EcPointMap for ScaleXPointMap {
    fn map(&self) {
        todo!()
    }
}