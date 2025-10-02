mod general_digest;
mod test_digest;
mod sha1_digest;
mod sha256_digest;
mod md2_digest;

pub use sha1_digest::Sha1Digest;
pub use sha256_digest::Sha256Digest;
pub use md2_digest::Md2Digest;