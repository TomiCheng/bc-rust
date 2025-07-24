mod ec_multiplier;
mod abstract_ec_multiplier;
mod fixed_point_comb_multiplier;

// Re-export the multiplier module
pub use ec_multiplier::EcMultiplier;
pub use abstract_ec_multiplier::AbstractEcMultiplier;