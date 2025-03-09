use crate::asn1::DerIntegerImpl;

/// The TbsCertificate object.
/// ```text
/// TbsCertificate ::= Sequence {
///   version          [ 0 ]  Version DEFAULT v1(0),
///   serialNumber            CertificateSerialNumber,
///   signature               AlgorithmIdentifier,
///   issuer                  Name,
///   validity                Validity,
///   subject                 Name,
///   subjectPublicKeyInfo    SubjectPublicKeyInfo,
///   issuerUniqueID    [ 1 ] IMPLICIT UniqueIdentifier OPTIONAL,
///   subjectUniqueID   [ 2 ] IMPLICIT UniqueIdentifier OPTIONAL,
///   extensions        [ 3 ] Extensions OPTIONAL
/// }
/// ```
pub struct TbsCertificateStructure {
    version: DerIntegerImpl,
    serial_number: DerIntegerImpl,
}