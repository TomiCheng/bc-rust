use crate::asn1::asn1_utilities::read_optional_context_iter;
use crate::asn1::x509::{AlgorithmIdentifier, SubjectPublicKeyInfo, Validity, X509Extensions, X509Name};
use crate::asn1::{Asn1BitString, Asn1Integer, Asn1Object, Asn1Sequence};
use crate::{BcError, Result};
use crate::asn1::try_from_tagged::TryFromTagged;

/// The TbsCertificate object.
/// ```text
/// TbsCertificate ::= Sequence {
///     version [0] Version DEFAULT v1(0),
///     serialNumber CertificateSerialNumber,
///     signature AlgorithmIdentifier,
///     issuer Name,
///     validity Validity,
///     subject Name,
///     subjectPublicKeyInfo SubjectPublicKeyInfo,
///     issuerUniqueID [1] IMPLICIT UniqueIdentifier OPTIONAL,
///     subjectUniqueID [2] IMPLICIT UniqueIdentifier OPTIONAL,
///     extensions [3] Extensions OPTIONAL
/// }
/// ```
/// Note: issuerUniqueID and subjectUniqueID are both deprecated by the IETF. This class
/// will parse them, but you really shouldn't be creating new ones.
pub struct TbsCertificateStructure {
    version: Asn1Integer,
    serial_number: Asn1Integer,
    signature: AlgorithmIdentifier,
    issuer: X509Name,
    validity: Validity,
    subject: X509Name,
    subject_public_key_info: SubjectPublicKeyInfo,
    issuer_unique_id: Option<Asn1BitString>,
    subject_unique_id: Option<Asn1BitString>,
    extensions: Option<X509Extensions>,
}

impl TbsCertificateStructure {
    pub fn new(
        version: Asn1Integer,
        serial_number: Asn1Integer,
        signature: AlgorithmIdentifier,
        issuer: X509Name,
        validity: Validity,
        subject: X509Name,
        subject_public_key_info: SubjectPublicKeyInfo,
        issuer_unique_id: Option<Asn1BitString>,
        subject_unique_id: Option<Asn1BitString>,
        extensions: Option<X509Extensions>,
    ) -> Self {
        TbsCertificateStructure {
            version,
            serial_number,
            signature,
            issuer,
            validity,
            subject,
            subject_public_key_info,
            issuer_unique_id,
            subject_unique_id,
            extensions,
        }
    }
    pub fn version(&self) -> u32 {
        &self.version.as_ref().as_u32() + 1
    }
    pub fn version_number(&self) -> &Asn1Integer {
        &self.version
    }
    pub fn serial_number(&self) -> &Asn1Integer {
        &self.serial_number
    }
    pub fn signature(&self) -> &AlgorithmIdentifier {
        &self.signature
    }
    pub fn issuer(&self) -> &X509Name {
        &self.issuer
    }
    pub fn validity(&self) -> &Validity {
        &self.validity
    }
    pub fn subject(&self) -> &X509Name {
        &self.subject
    }
    pub fn subject_public_key_info(&self) -> &SubjectPublicKeyInfo {
        &self.subject_public_key_info
    }
    pub fn issuer_unique_id(&self) -> Option<&Asn1BitString> {
        self.issuer_unique_id.as_ref()
    }
    pub fn subject_unique_id(&self) -> Option<&Asn1BitString> {
        self.subject_unique_id.as_ref()
    }
    pub fn extensions(&self) -> &Option<X509Extensions> {
        &self.extensions
    }
}
impl TryFrom<Asn1Object> for TbsCertificateStructure {
    type Error = BcError;

    fn try_from(asn1_object: Asn1Object) -> Result<Self> {
        if asn1_object.is_sequence() {
            let sequence: Asn1Sequence = asn1_object.try_into()?;
            if sequence.len() < 6 || sequence.len() > 10 {
                return Err(BcError::with_invalid_argument(format!("Bad sequence size: {}", sequence.len())));
            }

            let vector: Vec<Asn1Object> = sequence.into();
            let mut iter = vector.into_iter().peekable();
            let version = read_optional_context_iter(&mut iter, 0, true, Asn1Integer::try_from_tagged)?.unwrap_or(Asn1Integer::with_i64(0));
            let mut is_v1 = false;
            let mut is_v2 = false;
            if version.as_ref().as_u32() == 0 {
                is_v1 = true;
            } else if version.as_ref().as_u32() == 1 {
                is_v2 = true;
            } else if version.as_ref().as_u32() > 2 {
                return Err(BcError::with_invalid_argument(format!("version number not recognised: {}", version)));
            }

            let serial_number = iter.next().unwrap().try_into()?;
            let signature = iter.next().unwrap().try_into()?;
            let issuer = iter.next().unwrap().try_into()?;
            let validity = iter.next().unwrap().try_into()?;
            let subject = iter.next().unwrap().try_into()?;
            let subject_public_key_info = iter.next().unwrap().try_into()?;

            let mut issuer_unique_id = None;
            let mut subject_unique_id = None;
            let mut extensions = None;
            if !is_v1 {
                issuer_unique_id = read_optional_context_iter(&mut iter, 1, true, Asn1BitString::try_from_tagged)?;
                subject_unique_id = read_optional_context_iter(&mut iter, 2, true, Asn1BitString::try_from_tagged)?;
                if !is_v2 {
                    extensions = read_optional_context_iter(&mut iter, 3, true, X509Extensions::try_from_tagged)?;
                }
            }
            Ok(Self::new(
                version,
                serial_number,
                signature,
                issuer,
                validity,
                subject,
                subject_public_key_info,
                issuer_unique_id,
                subject_unique_id,
                extensions,
            ))
        } else {
            Err(BcError::with_invalid_cast("Expected a sequence for TbsCertificateStructure"))
        }
    }
}