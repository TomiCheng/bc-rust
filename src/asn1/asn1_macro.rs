#[macro_export]
macro_rules! define_oid {
    ($name:ident, $oid:expr, $doc:literal) => {
        #[doc = $doc]
        pub static $name: std::sync::LazyLock<crate::asn1::Asn1ObjectIdentifier> = std::sync::LazyLock::new(|| crate::asn1::Asn1ObjectIdentifier::with_str($oid).unwrap());
    };
    ($name:ident, $base:expr, $branch:expr, $doc:literal) => {
        #[doc = $doc]
        pub static $name: std::sync::LazyLock<crate::asn1::Asn1ObjectIdentifier> = std::sync::LazyLock::new(|| $base.branch($branch).unwrap());
    };
}