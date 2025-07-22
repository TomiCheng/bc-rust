use crate::{BcError, Result};
use crate::math::ec::FpCurve;
use crate::math::ec::fp_field_element::FpFieldElement;
use std::sync::{Arc, Weak};
// TODO

/// Elliptic curve points over Fp
pub struct FpPoint {
    curve: Weak<FpCurve>,
    x: Option<FpFieldElement>,
    y: Option<FpFieldElement>,
}

impl FpPoint {
    pub(crate) fn with_curve(curve: Weak<FpCurve>) -> Arc<Self> {
        Arc::new(FpPoint { curve, x: None, y: None })
    }
    pub(crate) fn with_curve_x_y(
        curve: Weak<FpCurve>,
        x: Option<FpFieldElement>,
        y: Option<FpFieldElement>,
    ) -> Result<Arc<Self>> {
        if x.is_none() != y.is_none() {
            return Err(BcError::with_invalid_argument("Exactly one of the field elements is null"));
        }
        Ok(Arc::new(FpPoint { curve, x, y }))
    }
    pub fn is_infinity(&self) -> bool {
        self.x.is_none() && self.y.is_none()
    }

    pub fn add(self: &Arc<FpPoint>, b: &Arc<FpPoint>) -> Arc<FpPoint> {
        if self.is_infinity() {
            return b.clone();
        }
        if b.is_infinity() {
            return self.clone();
        }
        if self == b {
            //return self.twice();
        }
        todo!();
    }
}
impl PartialEq for FpPoint {
    fn eq(&self, other: &Self) -> bool {
        match (&self.x, &self.y, &other.x, &other.y) {
            (Some(x1), Some(y1), Some(x2), Some(y2)) => x1 == x2 && y1 == y2,
            (None, None, None, None) => true,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {}
