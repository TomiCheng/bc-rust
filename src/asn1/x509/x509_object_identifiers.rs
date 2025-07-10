use crate::define_oid;

define_oid!(ATTRIBUTE_TYPE, "2.5.4", "X.500 attribute type base OID");
define_oid!(COMMON_NAME, ATTRIBUTE_TYPE, "3", "common name - StringType(SIZE(1..64))");
define_oid!(SURNAME, ATTRIBUTE_TYPE, "4", "surname");
define_oid!(SERIAL_NUMBER, ATTRIBUTE_TYPE, "5", "device serial number name - StringType(SIZE(1..64))");
define_oid!(COUNTRY_NAME, ATTRIBUTE_TYPE, "6", "country name - StringType(SIZE(2))");
define_oid!(LOCALITY_NAME, ATTRIBUTE_TYPE, "7", "locality name - StringType(SIZE(1..64))");
define_oid!(
    STATE_OR_PROVINCE_NAME,
    ATTRIBUTE_TYPE,
    "8",
    "state, or province name - StringType(SIZE(1..64))"
);
define_oid!(STREET, ATTRIBUTE_TYPE, "9", "street - StringType(SIZE(1..64))");
define_oid!(ORGANIZATION_NAME, ATTRIBUTE_TYPE, "10", "organization - StringType(SIZE(1..64))");
define_oid!(
    ORGANIZATIONAL_UNIT_NAME,
    ATTRIBUTE_TYPE,
    "11",
    "organizational unit name - StringType(SIZE(1..64))"
);
define_oid!(TITLE, ATTRIBUTE_TYPE, "12", "title");
define_oid!(DESCRIPTION, ATTRIBUTE_TYPE, "13", "description");
define_oid!(SEARCH_GUIDE, ATTRIBUTE_TYPE, "14", "search guide");
define_oid!(BUSINESS_CATEGORY, ATTRIBUTE_TYPE, "15", "businessCategory - DirectoryString(SIZE(1..128)");
define_oid!(POSTAL_ADDRESS, ATTRIBUTE_TYPE, "16", "postal address");
define_oid!(POSTAL_CODE, ATTRIBUTE_TYPE, "17", "postal code - DirectoryString(SIZE(1..40)");
define_oid!(TELEPHONE_NUMBER, ATTRIBUTE_TYPE, "20", "telephone number");
define_oid!(NAME, ATTRIBUTE_TYPE, "41", "Name");
define_oid!(ID_AT_NAME, ATTRIBUTE_TYPE, "41", "Name");
define_oid!(GIVEN_NAME, ATTRIBUTE_TYPE, "42", "given name");
define_oid!(INITIALS, ATTRIBUTE_TYPE, "43", "initials");
define_oid!(GENERATION, ATTRIBUTE_TYPE, "44", "generation");
define_oid!(UNIQUE_IDENTIFIER, ATTRIBUTE_TYPE, "45", "unique identifier");
define_oid!(DN_QUALIFIER, ATTRIBUTE_TYPE, "46", "DN qualifier");
define_oid!(PSEUDONYM, ATTRIBUTE_TYPE, "65", "pseudonym");
define_oid!(ROLE, ATTRIBUTE_TYPE, "72", "role");
define_oid!(ORGANIZATION_IDENTIFIER, ATTRIBUTE_TYPE, "97", "");

define_oid!(ID_SHA1, "1.3.14.3.2.26", "");
define_oid!(RIPE_MD160, "1.3.36.3.2.1", "");
define_oid!(RIPE_MD160_WITH_RSA_ENCRYPTION, "1.3.36.3.3.1.2", "");
define_oid!(ID_EARSA, "2.5.8.1.1", "");
define_oid!(ID_PKIX, "1.3.6.1.5.5.7", "");

define_oid!(ID_PE, ID_PKIX, "1", "");
define_oid!(PKIX_ALGORITHMS, ID_PKIX, "6", "");
define_oid!(ID_RSASSA_PSS_SHAKE128, PKIX_ALGORITHMS, "30", "");
define_oid!(ID_RSASSA_PSS_SHAKE256, PKIX_ALGORITHMS, "31", "");
define_oid!(ID_ECDSA_WITH_SHAKE128, PKIX_ALGORITHMS, "32", "");
define_oid!(ID_ECDSA_WITH_SHAKE256, PKIX_ALGORITHMS, "33", "");

define_oid!(ID_PDA, ID_PKIX, "9", "");

define_oid!(DATE_OF_BIRTH, ID_PDA, "1", "RFC 3039 DateOfBirth - GeneralizedTime - YYYYMMDD000000Z");
define_oid!(PLACE_OF_BIRTH, ID_PDA, "2", "RFC 3039 PlaceOfBirth - DirectoryString(SIZE(1..128))");
define_oid!(
    GENDER,
    ID_PDA,
    "3",
    "RFC 3039 DateOfBirth - PrintableString (SIZE(1)) -- \"M\", \"F\", \"m\" or \"f\""
);
define_oid!(
    COUNTRY_OF_CITIZENSHIP,
    ID_PDA,
    "4",
    "RFC 3039 CountryOfCitizenship - PrintableString (SIZE (2)) -- ISO 3166"
);
define_oid!(
    COUNTRY_OF_RESIDENCE,
    ID_PDA,
    "5",
    "RFC 3039 CountryOfResidence - PrintableString (SIZE (2)) -- ISO 3166"
);

define_oid!(ID_AD, ID_PKIX, "48", "");
define_oid!(ID_AD_OCSP, ID_AD, "1", "");
define_oid!(ID_AD_CA_ISSUERS, ID_AD, "2", "");

define_oid!(OCSP_ACCESS_METHOD, ID_AD, "1", "");
define_oid!(CRL_ACCESS_METHOD, ID_AD, "1", "");

define_oid!(ID_CE, "2.5.29", "");
define_oid!(NAME_AT_BIRTH, "1.3.36.8.3.14", "ISIS-MTT NameAtBirth - DirectoryString(SIZE(1..64)");
define_oid!(DC, "0.9.2342.19200300.100.1.25", "others");
define_oid!(UID, "0.9.2342.19200300.100.1.25", "LDAP User id.");
define_oid!(
    JURISDICTION_L,
    "1.3.6.1.4.1.311.60.2.1.1",
    "CA/Browser Forum https://cabforum.org/uploads/CA-Browser-Forum-BR-v2.0.0.pdf, Table 78"
);
define_oid!(
    JURISDICTION_ST,
    "1.3.6.1.4.1.311.60.2.1.2",
    "CA/Browser Forum https://cabforum.org/uploads/CA-Browser-Forum-BR-v2.0.0.pdf, Table 78"
);
define_oid!(
    JURISDICTION_C,
    "1.3.6.1.4.1.311.60.2.1.3",
    "CA/Browser Forum https://cabforum.org/uploads/CA-Browser-Forum-BR-v2.0.0.pdf, Table 78"
);
