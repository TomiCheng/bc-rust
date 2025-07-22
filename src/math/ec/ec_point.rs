
pub trait EcPoint {}

// TODO

mod tests {
    use crate::math::BigInteger;
    use crate::math::ec::FpCurve;
    use std::sync::Arc;
    use crate::math::ec::FpPoint;

    struct Fp {
        curve: Arc<FpCurve>,
        infinity: Arc<FpPoint>,
        points: Vec<Arc<FpPoint>>
    }

    impl Fp {
        pub fn new() -> Self {
            let q = BigInteger::with_string("1063").unwrap();
            let a = BigInteger::with_string("4").unwrap();
            let b = BigInteger::with_string("20").unwrap();
            let n = BigInteger::with_string("38").unwrap();
            let h = BigInteger::with_string("1").unwrap();
            let curve = FpCurve::new(q, a, b, n, h).unwrap();
            let infinity = curve.infinity();

            let point_source = [1, 5, 4, 10, 234, 1024, 817, 912];
            let mut points = Vec::new();
            for i in (0..point_source.len()).step_by(2) {
                let x = BigInteger::with_i32(point_source[i]);
                let y = BigInteger::with_i32(point_source[i + 1]);
                let point = curve.create_point(x, y).unwrap();
                points.push(point);
            }
            Fp { curve, infinity, points }
        }
    }

    #[test]
    fn test_add() {
        let fp = Fp::new();
        impl_test_add(&fp.points, fp.infinity);
    }

    fn impl_test_add(points: &[Arc<FpPoint>], infinity: Arc<FpPoint>) {
        assert_point_equal(&points[2], points[0].add(&points[1]));
    }
    fn assert_point_equal(point1: &Arc<FpPoint>, point2: Arc<FpPoint>) {

    }
}