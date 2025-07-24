use crate::math::BigInteger;
use crate::math::ec::{EcCurveRc, ScaleYNegateXPointMap};
use crate::math::ec::EcPointMapRc;
use crate::math::ec::morphism::GlvTypeAParameters;
use crate::math::ec::morphism::ec_endomorphism::EcEndomorphism;
use crate::math::ec::morphism::endomorphism_utilities::decompose_scalar;
use crate::math::ec::morphism::glv_endomorphism::GlvEndomorphism;
use std::sync::Arc;

pub struct GlvTypeAEndomorphism {
    parameters: GlvTypeAParameters,
    point_map: EcPointMapRc,
}

impl GlvTypeAEndomorphism {
    pub fn new(curve: EcCurveRc, parameters: GlvTypeAParameters) -> Self {
        let field_element = curve.create_field_element_from_big_integer(parameters.i());
        let point_map = ScaleYNegateXPointMap::new(field_element);
        GlvTypeAEndomorphism {
            parameters,
            point_map: Arc::new(point_map),
        }
    }
}

impl GlvEndomorphism for GlvTypeAEndomorphism {
    fn decompose_scalar(&self, k: &BigInteger) -> [BigInteger; 2] {
        decompose_scalar(self.parameters.split_params(), k)
    }
}

impl EcEndomorphism for GlvTypeAEndomorphism {
    fn point_map(&self) -> EcPointMapRc {
        self.point_map.clone()
    }

    fn has_efficient_point_map(&self) -> bool {
        true
    }
}
