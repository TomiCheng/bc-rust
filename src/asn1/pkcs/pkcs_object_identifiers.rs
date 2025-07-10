use crate::define_oid;

define_oid!(PKCS1, "1.2.840.113549.1.1", "iso(1) member-body(2) us(840) rsadsi(113549) pkcs(1) 1");
define_oid!(RSA_EXCRYPTION, PKCS1, "1", "");

define_oid!(PKCS9, "1.2.840.113549.1.9", "iso(1) member-body(2) us(840) rsadsi(113549) pkcs(1) 9");
define_oid!(PKCS9_AT_EMAIL_ADDRESS, PKCS9, "1", "");
define_oid!(PKCS9_AT_UNSTRUCTURED_NAME, PKCS9, "2", "");
define_oid!(PKCS9_AT_CONTENT_TYPE, PKCS9, "3", "");
define_oid!(PKCS9_AT_MESSAGE_DIGEST, PKCS9, "4", "");
define_oid!(PKCS9_AT_SIGNING_TIME, PKCS9, "5", "");
define_oid!(PKCS9_AT_COUNTER_SIGNATURE, PKCS9, "6", "");
define_oid!(PKCS9_AT_CHALLENGE_PASSWORD, PKCS9, "7", "");
define_oid!(PKCS9_AT_UNSTRUCTURED_ADDRESS, PKCS9, "8", "");
define_oid!(PKCS9_AT_EXTENDED_CERTIFICATE_ATTRIBUTES, PKCS9, "9", "");
define_oid!(PKCS9_AT_SIGNING_DESCRIPTION, PKCS9, "13", "");
define_oid!(PKCS9_AT_EXTENSION_REQUEST, PKCS9, "14", "");
define_oid!(PKCS9_AT_SMIME_CAPABILITIES, PKCS9, "15", "");
define_oid!(ID_SMIME, PKCS9, "16", "");

define_oid!(PKCS9_AT_FRIENDLY_NAME, PKCS9, "20", "");
define_oid!(PKCS9_AT_LOCAL_KEY_ID, PKCS9, "21", "");
