use crate::math::BigInteger;
use crate::math::ec::EcCurveRc;
use crate::math::ec::EcPointMapRc;
use crate::math::ec::ScaleXPointMap;
use crate::math::ec::morphism::GlvTypeBParameters;
use crate::math::ec::morphism::ec_endomorphism::EcEndomorphism;
use crate::math::ec::morphism::endomorphism_utilities::decompose_scalar;
use crate::math::ec::morphism::glv_endomorphism::GlvEndomorphism;
use std::sync::Arc;

pub struct GlvTypeBEndomorphism {
    parameters: GlvTypeBParameters,
    point_map: EcPointMapRc,
}

impl GlvTypeBEndomorphism {
    pub fn new(curve: EcCurveRc, parameters: GlvTypeBParameters) -> Self {
        let field_element = curve.create_field_element_from_big_integer(parameters.beta());
        let point_map = ScaleXPointMap::new(field_element);
        GlvTypeBEndomorphism {
            parameters,
            point_map: Arc::new(point_map),
        }
    }
}

impl GlvEndomorphism for GlvTypeBEndomorphism {
    fn decompose_scalar(&self, k: &BigInteger) -> [BigInteger; 2] {
        decompose_scalar(self.parameters.split_params(), k)
    }
}

impl EcEndomorphism for GlvTypeBEndomorphism {
    fn point_map(&self) -> EcPointMapRc {
        self.point_map.clone()
    }

    fn has_efficient_point_map(&self) -> bool {
        true
    }
}
