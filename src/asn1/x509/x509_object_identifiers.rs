use std::sync;
use crate::asn1;

macro_rules! define_oid {
    ($name:ident, $oid:expr) => {
        pub static $name: sync::LazyLock<asn1::Asn1ObjectIdentifier> =
        sync::LazyLock::new(|| asn1::Asn1ObjectIdentifier::with_str($oid).unwrap());
    };
    ($name:ident, $base:expr, $branch:expr) => {
        pub static $name: sync::LazyLock<asn1::Asn1ObjectIdentifier> =
        sync::LazyLock::new(|| $base.branch($branch).unwrap());
    };
}

define_oid!(ATTRIBUTE_TYPE, "2.5.4");
define_oid!(COMMON_NAME, ATTRIBUTE_TYPE, "3");
define_oid!(COUNTRY_NAME, ATTRIBUTE_TYPE, "6");
define_oid!(LOCALITY_NAME, ATTRIBUTE_TYPE, "7");
define_oid!(STATE_OR_PROVINCE_NAME, ATTRIBUTE_TYPE, "8");
define_oid!(ORGANIZATION_NAME, ATTRIBUTE_TYPE, "10");
define_oid!(ORGANIZATIONAL_UNIT_NAME, ATTRIBUTE_TYPE, "11");
define_oid!(ID_AT_TELEPHONE_NUMBER, ATTRIBUTE_TYPE, "20");
define_oid!(ID_AT_NAME, ATTRIBUTE_TYPE, "41");
define_oid!(ID_AT_ORGANIZATION_IDENTIFIER, ATTRIBUTE_TYPE, "97");


define_oid!(ID_SHA1, "1.3.14.3.2.26");
define_oid!(RIPE_MD160, "1.3.36.3.2.1");
define_oid!(RIPE_MD160_WITH_RSA_ENCRYPTION, "1.3.36.3.3.1.2");
define_oid!(ID_EARSA, "2.5.8.1.1");
define_oid!(ID_PKIX, "1.3.6.1.5.5.7");

define_oid!(ID_PE, ID_PKIX, "1");
define_oid!(PKIX_ALGORITHMS, ID_PKIX, "6");
define_oid!(ID_RSASSA_PSS_SHAKE128, PKIX_ALGORITHMS, "30");
define_oid!(ID_RSASSA_PSS_SHAKE256, PKIX_ALGORITHMS, "31");
define_oid!(ID_ECDSA_WITH_SHAKE128, PKIX_ALGORITHMS, "32");
define_oid!(ID_ECDSA_WITH_SHAKE256, PKIX_ALGORITHMS, "33");

define_oid!(ID_PDA, ID_PKIX, "9");

define_oid!(ID_AD, ID_PKIX, "48");
define_oid!(ID_AD_OCSP, ID_AD, "1");
define_oid!(ID_AD_CA_ISSUERS, ID_AD, "2");

define_oid!(OCSP_ACCESS_METHOD, ID_AD, "1");
define_oid!(CRL_ACCESS_METHOD, ID_AD, "1");

define_oid!(ID_CE, "2.5.29");