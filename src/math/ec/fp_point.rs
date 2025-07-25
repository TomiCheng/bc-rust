use std::cell::RefCell;
use crate::math::big_integer::{ONE, THREE};
use crate::math::ec::{CoordinateSystem, FpCurve, FpFieldElement};
use crate::{BcError, Result};
use rand::rng;
use std::rc::{Rc, Weak};

/// Elliptic curve points over Fp
pub struct FpPoint {
    curve: Weak<FpCurve>,
    x: Option<FpFieldElement>,
    y: Option<FpFieldElement>,
    zs: Vec<Option<FpFieldElement>>,
}

impl FpPoint {
    pub(crate) fn with_curve(curve: Weak<FpCurve>) -> Result<Rc<Self>> {
        let zs = initial_z_coordinates(curve.upgrade())?;
        Ok(Rc::new(FpPoint {
            curve,
            x: None,
            y: None,
            zs,
        }))
    }

    pub(crate) fn with_curve_x_y(
        curve: Weak<FpCurve>,
        x: Option<FpFieldElement>,
        y: Option<FpFieldElement>,
    ) -> Result<Rc<Self>> {
        if x.is_none() != y.is_none() {
            return Err(BcError::with_invalid_argument(
                "Exactly one of the field elements is null",
            ));
        }
        let zs = initial_z_coordinates(curve.upgrade())?;
        Ok(Rc::new(FpPoint { curve, x, y, zs }))
    }
    pub(crate) fn with_curve_x_y_zs(
        curve: Weak<FpCurve>,
        x: Option<FpFieldElement>,
        y: Option<FpFieldElement>,
        zs: Vec<Option<FpFieldElement>>,
    ) -> Result<Rc<Self>> {
        if x.is_none() != y.is_none() {
            return Err(BcError::with_invalid_argument(
                "Exactly one of the field elements is null",
            ));
        }
        Ok(Rc::new(FpPoint { curve, x, y, zs }))
    }
    pub fn curve(&self) -> Option<Rc<FpCurve>> {
        self.curve.upgrade()
    }
    pub fn is_infinity(&self) -> bool {
        self.x.is_none() && self.y.is_none()
    }
    pub(crate) fn raw_x_coordinate(&self) -> Option<&FpFieldElement> {
        self.x.as_ref()
    }
    pub(crate) fn raw_y_coordinate(&self) -> Option<&FpFieldElement> {
        self.y.as_ref()
    }
    pub fn get_z_coordinates_with_index(&self, index: usize) -> Option<&FpFieldElement> {
        if index >= self.zs.len() {
            None
        } else {
            self.zs[index].as_ref()
        }
    }
    pub fn normalize(self: Rc<Self>) -> Result<Rc<Self>> {
        if self.is_infinity() {
            return Ok(self.clone());
        }

        match self.curve_coordinate_system() {
            CoordinateSystem::Affine | CoordinateSystem::LambdaAffine => Ok(self.clone()),
            _ => {
                let z = self.zs[0].as_ref().unwrap();
                if z.is_one() {
                    return Ok(self.clone());
                }

                if let Some(curve) = self.curve.upgrade() {
                    /*
                     * Use blinding to avoid the side-channel leak identified and analyzed in the paper
                     * "Yet another GCD based inversion side-channel affecting ECC implementations" by Nir
                     * Drucker and Shay Gueron.
                     *
                     * To blind the calculation of z^-1, choose a multiplicative (i.e. non-zero) field
                     * element 'b' uniformly at random, then calculate the result instead as (z * b)^-1 * b.
                     * Any side-channel in the implementation of 'inverse' now only leaks information about
                     * the value (z * b), and no longer reveals information about 'z' itself.
                     */
                    let mut rng = rng();
                    let b = curve.random_field_element_multiply(&mut rng)?;
                    let z_inv = z.multiply(&b)?.invert()?.multiply(&b)?;
                    self.normalize_with_z_inv(z_inv)
                } else {
                    Err(BcError::with_invalid_operation(
                        "Detached points must be in affine coordinates",
                    ))
                }
            }
        }
    }
    pub(crate) fn normalize_with_z_inv(self: Rc<Self>, z_inv: FpFieldElement) -> Result<Rc<Self>> {
        match self.curve_coordinate_system() {
            CoordinateSystem::Homogeneous | CoordinateSystem::LambdaProjective => {
                self.create_scaled_point(&z_inv, &z_inv)
            }
            CoordinateSystem::Jacobian
            | CoordinateSystem::JacobianChudnovsky
            | CoordinateSystem::JacobianModified => {
                let z_inv2 = z_inv.square()?;
                let z_inv3 = z_inv2.multiply(&z_inv)?;
                self.create_scaled_point(&z_inv2, &z_inv3)
            }
            _ => Err(BcError::with_invalid_operation(
                "not a projective coordinate system",
            )),
        }
    }
    pub fn curve_coordinate_system(&self) -> CoordinateSystem {
        if let Some(curve) = self.curve() {
            curve.coordinate_system()
        } else {
            CoordinateSystem::Affine
        }
    }
    fn create_scaled_point(
        self: Rc<Self>,
        sx: &FpFieldElement,
        sy: &FpFieldElement,
    ) -> Result<Rc<FpPoint>> {
        let x = self.raw_x_coordinate().unwrap().multiply(sx)?;
        let y = self.raw_y_coordinate().unwrap().multiply(sy)?;
        self.curve
            .upgrade()
            .unwrap()
            .create_raw_point_x_y(Some(x), Some(y))
    }
    pub fn add(self: &mut Rc<FpPoint>, b: &Rc<FpPoint>) -> Result<Rc<Self>> {
        if self.is_infinity() {
            return Ok(b.clone());
        }
        if b.is_infinity() {
            return Ok(self.clone());
        }
        if Rc::ptr_eq(&self, &b) {
            return self.twice();
        }

        let curve = self
            .curve()
            .ok_or(BcError::with_invalid_operation("curve is None"))?;
        let coord = curve.coordinate_system();

        let x1 = self
            .raw_x_coordinate()
            .ok_or(BcError::with_invalid_operation("x1 is None"))?;
        let y1 = self
            .raw_y_coordinate()
            .ok_or(BcError::with_invalid_operation("y1 is None"))?;
        let x2 = b
            .raw_x_coordinate()
            .ok_or(BcError::with_invalid_operation("x2 is None"))?;
        let y2 = b
            .raw_y_coordinate()
            .ok_or(BcError::with_invalid_operation("y2 is None"))?;

        match coord {
            CoordinateSystem::Affine => {
                let dx = x2.subtract(&x1);
                let dy = y2.subtract(&y1);
                if dx.is_zero() {
                    if dy.is_zero() {
                        return self.twice();
                    }
                    return Ok(curve.infinity().clone());
                }

                let gamma = dy.divide(&dx)?;
                let x3 = gamma.square()?.subtract(&x1).subtract(&x2);
                let y3 = gamma.multiply(&x1.subtract(&x3))?.subtract(y1);

                return FpPoint::with_curve_x_y(Rc::downgrade(&curve), Some(x3), Some(y3));
            }
            CoordinateSystem::Homogeneous => {
                let z1 = self.zs[0].as_ref().unwrap();
                let z2 = b.zs[0].as_ref().unwrap();

                let z1_is_one = z1.is_one();
                let z2_is_one = z2.is_one();

                let u1 = if z1_is_one { y2 } else { &y2.multiply(&z1)? };
                let u2 = if z2_is_one { y1 } else { &y1.multiply(&z2)? };
                let u = u1.subtract(&u2);
                let v1 = if z1_is_one { x2 } else { &x2.multiply(&z1)? };
                let v2 = if z2_is_one { x1 } else { &x1.multiply(&z2)? };
                let v = v1.subtract(&v2);

                if v.is_zero() {
                    if u.is_zero() {
                        return self.twice();
                    }
                    return Ok(curve.infinity().clone());
                }

                let w = if z1_is_one {
                    z2
                } else if z2_is_one {
                    z1
                } else {
                    &z1.multiply(&z2)?
                };
                let v_squared = v.square()?;
                let v_cubed = v_squared.multiply(&v)?;
                let v_squared_v2 = v_squared.multiply(&v2)?;
                let a = u
                    .square()?
                    .multiply(w)?
                    .subtract(&v_cubed)
                    .subtract(&Self::two(&v_squared_v2));

                let x3 = v.multiply(&a)?;
                let y3 = v_squared_v2
                    .subtract(&a)
                    .multiply_minus_product(&u, &u2, &v_cubed)?;
                let z3 = Some(v_cubed.multiply(&w)?);

                FpPoint::with_curve_x_y_zs(Rc::downgrade(&curve), Some(x3), Some(y3), vec![z3])
            }
            CoordinateSystem::Jacobian | CoordinateSystem::JacobianModified => {
                let z1 = self.zs[0].as_ref().unwrap();
                let z2 = b.zs[0].as_ref().unwrap();

                let z1_is_one = z1.is_one();

                let x3;
                let y3;
                let mut z3;
                let mut z3_squared = None;

                if !z1_is_one && z1 == z2 {
                    let dx = x1.subtract(&x2);
                    let dy = y1.subtract(&y2);
                    if dx.is_zero() {
                        if dy.is_zero() {
                            return self.twice();
                        }
                        return Ok(curve.infinity().clone());
                    }

                    let c = dx.square()?;
                    let w1 = x1.multiply(&c)?;
                    let w2 = x2.multiply(&c)?;
                    let a1 = w1.subtract(&w2).multiply(&y1)?;

                    x3 = dy.square()?.subtract(&w1).subtract(&w2);
                    y3 = w1.subtract(&x3).multiply(&dy)?.subtract(&a1);
                    z3 = dx;

                    if z1_is_one {
                        z3_squared = Some(c);
                    } else {
                        z3 = z3.multiply(&z1)?;
                    }
                } else {
                    let z1_squared;
                    let u2;
                    let s2;
                    if z1_is_one {
                        z1_squared = z1.clone();
                        u2 = x2.clone();
                        s2 = y2.clone();
                    } else {
                        z1_squared = z1.square()?;
                        u2 = z1_squared.multiply(&x2)?;
                        let z1_cubed = z1_squared.multiply(&z1)?;
                        s2 = z1_cubed.multiply(&y2)?;
                    }

                    let z2_is_one = z2.is_one();
                    let z2_squared;
                    let u1;
                    let s1;

                    if z2_is_one {
                        z2_squared = z2.clone();
                        u1 = x1.clone();
                        s1 = y1.clone();
                    } else {
                        z2_squared = z2.square()?;
                        u1 = z2_squared.multiply(&x1)?;
                        let z2_cubed = z2_squared.multiply(&z2)?;
                        s1 = z2_cubed.multiply(&y1)?;
                    }

                    let h = u1.subtract(&u2);
                    let r = s1.subtract(&s2);

                    if h.is_zero() {
                        if r.is_zero() {
                            return self.twice();
                        }
                        return Ok(curve.infinity().clone());
                    }

                    let h_squared = h.square()?;
                    let g = h_squared.multiply(&h)?;
                    let v = h_squared.multiply(&u1)?;

                    x3 = r.square()?.add(&g).subtract(&Self::two(&v));
                    y3 = v.subtract(&x3).multiply_minus_product(&g, &g, &s1)?;
                    z3 = h.clone();

                    if !z1_is_one {
                        z3 = z3.multiply(&z1)?;
                    }
                    if !z2_is_one {
                        z3 = z3.multiply(&z2)?;
                    }
                    if z3 == h {
                        z3_squared = Some(h_squared);
                    }
                }

                let zs;
                if coord == CoordinateSystem::JacobianModified {
                    let w3 = self.calculate_jacobian_modified_w(&z3, z3_squared)?;
                    zs = vec![Some(z3), Some(w3)];
                } else {
                    zs = vec![Some(z3)];
                }

                FpPoint::with_curve_x_y_zs(Rc::downgrade(&curve), Some(x3), Some(y3), zs)
            }
            _ => Err(BcError::with_invalid_operation(
                "unsupported coordinate system",
            )),
        }
    }
    pub fn twice(self: &Rc<Self>) -> Result<Rc<Self>> {
        if self.is_infinity() {
            return Ok(self.clone());
        }

        let curve = self
            .curve()
            .ok_or(BcError::with_invalid_operation("curve is None"))?;
        let y1 = self
            .raw_y_coordinate()
            .ok_or(BcError::with_invalid_operation("y1 is None"))?;

        if y1.is_zero() {
            return Ok(curve.infinity().clone());
        }

        let coord = curve.coordinate_system();
        let x1 = self
            .raw_x_coordinate()
            .ok_or(BcError::with_invalid_operation("x1 is None"))?;

        match coord {
            CoordinateSystem::Affine => {
                let x1_squared = x1.square()?;
                let gamma = Self::three(&x1_squared).add(curve.a()).divide(&Self::two(&y1))?;
                let x3 = gamma.square()?.subtract(&Self::two(&x1));
                let y3 = gamma.multiply(&x1.subtract(&x3))?.subtract(&y1);
                FpPoint::with_curve_x_y(Rc::downgrade(&curve), Some(x3), Some(y3))
            },
            CoordinateSystem::Homogeneous => {
                let z1 = self.zs[0].as_ref().unwrap();
                let z1_is_one = z1.is_one();
                let mut w = curve.a().clone();
                if !w.is_zero() && z1_is_one {
                    w = w.multiply(&z1.square()?)?;
                }
                w = w.add(&Self::three(&x1.square()?));

                let s = if z1_is_one { y1.clone() } else { y1.multiply(&z1)? };
                let t = if z1_is_one { y1.clone() } else { x1.multiply(&y1)? };
                let b = x1.multiply(&t)?;
                let i4b = Self::four(&b);
                let h = w.square()?.subtract(&Self::two(&i4b));

                let i2s = Self::two(&s);
                let x3 = h.multiply(&i2s)?;
                let i2t = Self::two(&t);
                let y3 = i4b.subtract(&h).multiply(&w)?.subtract(&Self::two(&i2t.square()?));
                let i4s_squared = if z1_is_one { Self::two(&i2t) } else { i2s.square()? };
                let z3 = Self::two(&i4s_squared).multiply(&s)?;

                FpPoint::with_curve_x_y_zs(
                    Rc::downgrade(&curve),
                    Some(x3),
                    Some(y3),
                    vec![Some(z3)],
                )
            },
            CoordinateSystem::Jacobian => {
                let z1 = self.zs[0].as_ref().unwrap();

                let z1_is_one = z1.is_one();

                let y1_squared = y1.square()?;
                let t = y1_squared.square()?;

                let a4 = curve.a();
                let a4_neg = a4.negate();

                let mut m;
                let s;

                if a4_neg.to_big_integer() == *THREE {
                    let z1_squared = if z1_is_one { z1.clone() } else { z1.square()? };
                    m = Self::three(&x1.add(&z1_squared).multiply(&x1.subtract(&z1_squared))?);
                    s = Self::four(&y1_squared.multiply(&x1)?);
                } else {
                    let x1_squared = x1.square()?;
                    m = Self::three(&x1_squared);
                    if z1_is_one {
                        m = m.add(a4);
                    } else if !a4.is_zero() {
                        let z1_squared = if z1_is_one { z1.clone() } else { z1.square()? };
                        let z1_pow4 = z1_squared.square()?;
                        if a4_neg.bit_length() < a4.bit_length() {
                            m = m.subtract(&z1_pow4.multiply(&a4_neg)?);
                        } else {
                            m = m.add(&z1_pow4.multiply(&a4)?);
                        }
                    }
                    s = Self::four(&x1.multiply(&y1_squared)?);
                }

                let x3 = m.square()?.subtract(&Self::two(&s));
                let y3 = s.subtract(&x3).multiply(&m)?.subtract(&Self::eight(&t));
                let mut z3 = Self::two(&y1);
                if !z1_is_one {
                    z3 = z3.multiply(&z1)?;
                }
                FpPoint::with_curve_x_y_zs(Rc::downgrade(&curve), Some(x3), Some(y3), vec![Some(z3)])
            },
            CoordinateSystem::JacobianModified => {
                self.twice_jacobian_modified(true)
            },
            _ => {
                Err(BcError::with_invalid_operation("unsupported coordinate system"))
            }
        }
    }
    pub fn two(x: &FpFieldElement) -> FpFieldElement {
        x.add(x)
    }
    pub fn three(x: &FpFieldElement) -> FpFieldElement {
        Self::two(&x).add(&x)
    }
    pub fn four(x: &FpFieldElement) -> FpFieldElement {
        Self::two(&Self::two(x))
    }
    fn eight(x: &FpFieldElement) -> FpFieldElement {
        Self::four(&Self::two(&x))
    }
    fn calculate_jacobian_modified_w(
        &self,
        z: &FpFieldElement,
        z_squared: Option<FpFieldElement>,
    ) -> Result<FpFieldElement> {
        let curve = self
            .curve()
            .ok_or(BcError::with_invalid_operation("curve is None"))?;

        let a4 = curve.a();
        if a4.is_zero() || z.is_one() {
            return Ok(a4.clone());
        }

        let z_squared = if z_squared.is_none() {
            z.square()?
        } else {
            z_squared.unwrap()
        };

        let mut w = z_squared.square()?;
        let a4_neg = a4.negate();
        if a4_neg.bit_length() < a4.bit_length() {
            w = w.multiply(&a4_neg)?.negate();
        } else {
            w = w.multiply(&a4)?;
        }
        Ok(w)
    }
    fn twice_jacobian_modified(&self, calculate_w: bool) -> Result<Rc<FpPoint>> {
        let x1 = self.raw_x_coordinate();
        let y1 = self.raw_y_coordinate();
        let z1 = &self.zs[0];
        let w1 = self.get_jacobian_modified_w()?;
        todo!();
    }
    fn get_jacobian_modified_w(&self) -> Result<FpFieldElement> {
        let z1 = self.zs[1].as_ref();
        if z1.is_none() {
            let z0 = self.zs[0].as_ref().unwrap();
            let z1 = self.calculate_jacobian_modified_w(&z0, None)?;
            let z2 = z1.clone();
            //let mut zs_mut = self.zs.borrow_mut();

            //self.zs[1] = Some(z1);
            //return Ok(z2);

            todo!();
        } else {
            return Ok(z1.unwrap().clone());
        }
    }
    // fn calculate_jacobian_modified_w(&self, z: &FpFieldElement, z_squared: Option<&FpFieldElement>) -> Result<FpFieldElement> {
    //     let curve = self
    //         .curve()
    //         .ok_or(BcError::with_invalid_operation("curve is None"))?;
    //
    //     let a4 = curve.a();
    //     if a4.is_zero() || z.is_one() {
    //         return Ok(a4.clone());
    //     }
    //
    //     let z_squared = if z_squared.is_none() {
    //         z.square()?
    //     } else {
    //         z_squared.unwrap().clone()
    //     };
    //
    //     let mut w = z_squared.square()?;
    //     let a4_neg = a4.negate();
    //     if a4_neg.bit_length() < a4.bit_length() {
    //         w = w.multiply(&a4_neg)?.negate();
    //     } else {
    //         w = w.multiply(&a4)?;
    //     }
    //     Ok(w)
    // }
}

fn initial_z_coordinates(curve: Option<Rc<FpCurve>>) -> Result<Vec<Option<FpFieldElement>>> {
    if let Some(curve) = curve {
        match curve.coordinate_system() {
            CoordinateSystem::Homogeneous
            | CoordinateSystem::Jacobian
            | CoordinateSystem::LambdaProjective => {
                return Ok(vec![Some(curve.create_field_element((*ONE).clone())?)]);
            }
            CoordinateSystem::JacobianChudnovsky => {
                return Ok(vec![
                    Some(curve.create_field_element((*ONE).clone())?),
                    Some(curve.create_field_element((*ONE).clone())?),
                    Some(curve.create_field_element((*ONE).clone())?),
                ]);
            }
            CoordinateSystem::JacobianModified => {
                return Ok(vec![
                    Some(curve.create_field_element((*ONE).clone())?),
                    Some(curve.a().clone()),
                ]);
            }

            _ => {}
        }
    }
    Ok(vec![])
}
