#[macro_export]
macro_rules! define_oid {
    ($name:ident, $oid:expr) => {
        pub static $name: LazyLock<Asn1ObjectIdentifier> = LazyLock::new(|| Asn1ObjectIdentifier::with_str($oid).unwrap());
    };
    ($name:ident, $oid:expr, $doc:literal) => {
        #[doc = $doc]
        pub static $name: LazyLock<Asn1ObjectIdentifier> = LazyLock::new(|| Asn1ObjectIdentifier::with_str($oid).unwrap());
    };
    ($name:ident, $base:expr, $branch:expr) => {
        pub static $name: LazyLock<Asn1ObjectIdentifier> = LazyLock::new(|| $base.branch($branch).unwrap());
    };
    ($name:ident, $base:expr, $branch:expr, $doc:literal) => {
        #[doc = $doc]
        pub static $name: LazyLock<Asn1ObjectIdentifier> = LazyLock::new(|| $base.branch($branch).unwrap());
    };
}