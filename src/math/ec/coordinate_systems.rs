#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinateSystem {
    Affine = 0,
    Homogeneous = 1,
    Jacobian = 2,
    JacobianChudnovsky = 3,
    JacobianModified = 4,
    LambdaAffine = 5,
    LambdaProjective = 6,
    Skewed = 7,
}
pub const ALL_COORDINATE_SYSTEMS: [CoordinateSystem; 8] = [
    CoordinateSystem::Affine,
    CoordinateSystem::Homogeneous,
    CoordinateSystem::Jacobian,
    CoordinateSystem::JacobianChudnovsky,
    CoordinateSystem::JacobianModified,
    CoordinateSystem::LambdaAffine,
    CoordinateSystem::LambdaProjective,
    CoordinateSystem::Skewed,
];