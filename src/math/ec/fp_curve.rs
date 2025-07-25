use crate::math::BigInteger;
use crate::math::ec::fp_field_element::calculate_residue;
use crate::math::ec::{CoordinateSystem, FpFieldElement, FpPoint};
use crate::math::field::{FiniteField, PrimeField, finite_fields};
use crate::util::big_integers;
use crate::{BcError, Result};
use rand::RngCore;
use std::rc::Rc;

/// Elliptic curve over Fp
pub struct FpCurve {
    field: PrimeField,
    a: FpFieldElement,
    b: FpFieldElement,
    order: BigInteger,
    cofactor: BigInteger,
    coordinate_system: CoordinateSystem,
    q: BigInteger,
    r: Option<BigInteger>,
    infinity: Rc<FpPoint>,
}

impl FpCurve {
    const FP_DEFAULT_COORDS: CoordinateSystem = CoordinateSystem::JacobianModified;

    pub fn new(
        q: BigInteger,
        a: BigInteger,
        b: BigInteger,
        order: BigInteger,
        cofactor: BigInteger,
    ) -> Result<Rc<Self>> {
        let field = finite_fields::create_prime_field(q.clone())?;
        let r = calculate_residue(&q)?;
        let a = create_field_element(q.clone(), r.as_ref().cloned(), a)?;
        let b = create_field_element(q.clone(), r.as_ref().cloned(), b)?;

        Ok(Rc::new_cyclic(|weak| FpCurve {
            field,
            a,
            b,
            order,
            cofactor,
            coordinate_system: Self::FP_DEFAULT_COORDS,
            q,
            r,
            infinity: FpPoint::with_curve(weak.clone()).unwrap(),
        }))
    }
    // getting
    pub fn q(&self) -> &BigInteger {
        &self.q
    }
    pub fn a(&self) -> &FpFieldElement {
        &self.a
    }
    pub fn b(&self) -> &FpFieldElement {
        &self.b
    }
    pub fn coordinate_system(&self) -> CoordinateSystem {
        self.coordinate_system
    }
    // override
    pub fn supports_coordinate_system(&self, coord: CoordinateSystem) -> bool {
        coord == CoordinateSystem::Affine
            || coord == CoordinateSystem::Homogeneous
            || coord == CoordinateSystem::Jacobian
            || coord == CoordinateSystem::JacobianModified
    }
    pub fn infinity(&self) -> Rc<FpPoint> {
        self.infinity.clone()
    }
    pub fn field_length(&self) -> usize {
        self.q.bit_length()
    }
    pub fn create_field_element(&self, x: BigInteger) -> Result<FpFieldElement> {
        create_field_element(self.q.clone(), self.r.clone(), x)
    }
    pub(crate) fn create_raw_point_x_y(
        self: &Rc<Self>,
        x: Option<FpFieldElement>,
        y: Option<FpFieldElement>,
    ) -> Result<Rc<FpPoint>> {
        Ok(FpPoint::with_curve_x_y(Rc::downgrade(&self), x, y)?)
    }
    pub(crate) fn create_raw_point_with_x_y_zs(
        self: Rc<Self>,
        x: Option<FpFieldElement>,
        y: Option<FpFieldElement>,
        zs: Vec<FpFieldElement>,
    ) -> Result<Rc<FpPoint>> {
        Ok(FpPoint::with_curve_x_y_zs(Rc::downgrade(&self), x, y, zs)?)
    }
    pub fn import_point(self: Rc<Self>, point: Rc<FpPoint>) -> Result<Rc<FpPoint>> {
        if let Some(other_curve) = point.curve() {
            if Rc::ptr_eq(&self, &other_curve)
                && self.coordinate_system() == CoordinateSystem::Jacobian
                && !point.is_infinity()
            {
                match other_curve.coordinate_system() {
                    CoordinateSystem::Jacobian
                    | CoordinateSystem::JacobianChudnovsky
                    | CoordinateSystem::JacobianModified => {
                        let x = point
                            .raw_x_coordinate()
                            .ok_or(BcError::with_invalid_operation("raw_x_coordinate is None"))?
                            .to_big_integer();
                        let y = point
                            .raw_y_coordinate()
                            .ok_or(BcError::with_invalid_operation("raw_y_coordinate is None"))?
                            .to_big_integer();
                        let z = point
                            .get_z_coordinates_with_index(0)
                            .ok_or(BcError::with_invalid_operation(
                                "get_z_coordinates_with_index is None",
                            ))?
                            .to_big_integer();

                        let x = self.create_field_element(x)?;
                        let y = self.create_field_element(y)?;
                        let zs = vec![self.create_field_element(z)?];

                        return Ok(FpPoint::with_curve_x_y_zs(
                            Rc::downgrade(&self),
                            Some(x),
                            Some(y),
                            zs,
                        )?);
                    }
                    _ => {}
                }
            }

            if Rc::ptr_eq(&self, &other_curve) {
                return Ok(point);
            }
        }
        if point.is_infinity() {
            return Ok(point);
        }

        let p = point.normalize()?;
        //Ok(self.create_point_x_y())

        todo!();
    }
    pub fn create_point_x_y(
        self: &Rc<Self>,
        x: Option<BigInteger>,
        y: Option<BigInteger>,
    ) -> Result<Rc<FpPoint>> {
        let x = x.map(|v| self.create_field_element(v)).transpose()?;
        let y = y.map(|v| self.create_field_element(v)).transpose()?;
        self.create_raw_point_x_y(x, y)
    }
    pub fn random_field_element_multiply<TRngCore: RngCore>(
        &self,
        r: &mut TRngCore,
    ) -> Result<FpFieldElement> {
        // NOTE: BigInteger comparisons in the rejection sampling are not constant-time, so we
        // use the product of two independent elements to mitigate side-channels.
        let p = self.field.characteristic();
        let fe1 = self.create_field_element(impl_random_field_element_multiply(r, p))?;
        let fe2 = self.create_field_element(impl_random_field_element_multiply(r, p))?;
        fe1.multiply(&fe2)
    }
}

fn create_field_element(
    q: BigInteger,
    r: Option<BigInteger>,
    x: BigInteger,
) -> Result<FpFieldElement> {
    if x.sign() < 0 || &x >= &q {
        return Err(BcError::with_invalid_argument(
            "value invalid for Fp field element",
        ));
    }
    Ok(FpFieldElement::new(q, r, x))
}

fn impl_random_field_element_multiply<TRngCore: RngCore>(
    r: &mut TRngCore,
    p: &BigInteger,
) -> BigInteger {
    let mut x: BigInteger;
    loop {
        x = big_integers::create_random_big_integer(p.bit_length(), r);
        if !(x.sign() <= 0 || &x >= p) {
            break;
        }
    }
    x
}

// // TODO
