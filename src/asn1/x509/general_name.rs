use crate::asn1::asn1_utilities::try_from_choice_tagged;
use crate::asn1::try_from_tagged::{TryFromTagged, TryIntoTagged};
use crate::asn1::x509::{EdiPartyName, X509Name};
use crate::asn1::{Asn1Ia5String, Asn1Object, Asn1ObjectIdentifier, Asn1OctetString, Asn1Sequence, Asn1TaggedObject};
use crate::{BcError, Result};

/// The GeneralName object.
/// ```text
/// GeneralName ::= CHOICE {
///     otherName [0] OtherName,
///     rfc822Name [1] IA5String,
///     dNSName [2] IA5String,
///     x400Address [3] ORAddress,
///     directoryName [4] Name,
///     ediPartyName [5] EDIPartyName,
///     uniformResourceIdentifier [6] IA5String,
///     ipAddress [7] OCTET STRING,
///     registeredID [8] OBJECT IDENTIFIER
/// }
///
/// OtherName ::= SEQUENCE {
///     type-id OBJECT IDENTIFIER,
///     value [0] EXPLICIT ANY DEFINED BY type-id
/// }
///
/// EDIPartyName ::= SEQUENCE {
///    nameAssigner [0] DirectoryString OPTIONAL,
///    partyName [1] DirectoryString
/// }
/// ```
pub enum GeneralName {
    OtherName(Asn1Sequence),
    Rfc822Name(Asn1Ia5String),
    DnsName(Asn1Ia5String),
    X400Address(Asn1Sequence),
    DirectoryName(X509Name),
    EdiPartyName(EdiPartyName),
    UniformResourceIdentifier(Asn1Ia5String),
    IpAddress(Asn1OctetString),
    RegisteredId(Asn1ObjectIdentifier),
}

impl GeneralName {
    pub const OTHER_NAME: u8 = 0;
    pub const RFC822_NAME: u8 = 1;
    pub const DNS_NAME: u8 = 2;
    pub const X400_ADDRESS: u8 = 3;
    pub const DIRECTORY_NAME: u8 = 4;
    pub const EDI_PARTY_NAME: u8 = 5;
    pub const UNIFORM_RESOURCE_IDENTIFIER: u8 = 6;
    pub const IP_ADDRESS: u8 = 7;
    pub const REGISTERED_ID: u8 = 8;

    pub fn with_directory_name(name: X509Name) -> Self {
        GeneralName::DirectoryName(name)
    }
    fn try_from_base_object(tagged_object: Asn1TaggedObject) -> Result<Self> {
        if tagged_object.has_context_tag() {
            match tagged_object.tag_no() {
                //Self::OTHER_NAME => Ok(GeneralName::OtherName(Asn1Sequence::get_tagged(tagged_object, false)?)),
                Self::RFC822_NAME => Ok(GeneralName::Rfc822Name(tagged_object.try_into_tagged(false)?)),
                Self::DNS_NAME => Ok(GeneralName::DnsName(tagged_object.try_into_tagged(false)?)),
                //Self::X400_ADDRESS => Ok(GeneralName::X400Address(Asn1Sequence::get_tagged(tagged_object, false)?)),
                Self::DIRECTORY_NAME => Ok(GeneralName::DirectoryName(X509Name::get_tagged(tagged_object, false)?)),
                //Self::EDI_PARTY_NAME => {
                //let sequence = Asn1Sequence::get_tagged(tagged_object, false)?;
                //Ok(GeneralName::EdiPartyName(EdiPartyName::from_sequence(sequence)?))
                //}
                Self::UNIFORM_RESOURCE_IDENTIFIER => Ok(GeneralName::UniformResourceIdentifier(tagged_object.try_into_tagged(false)?)),
                Self::IP_ADDRESS => Ok(GeneralName::IpAddress(tagged_object.try_into_tagged(false)?)),
                Self::REGISTERED_ID => Ok(GeneralName::RegisteredId(tagged_object.try_into_tagged(false)?)),
                _ => Err(BcError::with_invalid_argument(format!(
                    "Expected a context tag for GeneralName tag_no: {}",
                    tagged_object.tag_no()
                ))),
            }
        } else {
            Err(BcError::with_invalid_argument("Expected a context tag for GeneralName"))
        }
    }
}
impl From<GeneralName> for Asn1Object {
    fn from(value: GeneralName) -> Self {
        let (is_explicit, tag_no, object) = match value {
            GeneralName::OtherName(seq) => (false, GeneralName::OTHER_NAME, seq.into()),
            GeneralName::Rfc822Name(ia5) => (false, GeneralName::RFC822_NAME, ia5.into()),
            GeneralName::DnsName(ia5) => (false, GeneralName::DNS_NAME, ia5.into()),
            GeneralName::X400Address(seq) => (false, GeneralName::X400_ADDRESS, seq.into()),
            GeneralName::DirectoryName(name) => (true, GeneralName::DIRECTORY_NAME, name.into()),
            GeneralName::EdiPartyName(edi_party_name) => (true, GeneralName::EDI_PARTY_NAME, edi_party_name.into()),
            GeneralName::UniformResourceIdentifier(ia5) => (false, GeneralName::UNIFORM_RESOURCE_IDENTIFIER, ia5.into()),
            GeneralName::IpAddress(octet_string) => (false, GeneralName::IP_ADDRESS, octet_string.into()),
            GeneralName::RegisteredId(object_id) => (false, GeneralName::REGISTERED_ID, object_id.into()),
        };
        Asn1TaggedObject::from_explicit_tag_object(is_explicit, tag_no, object).into()
    }
}
impl TryFrom<Asn1Object> for GeneralName {
    type Error = BcError;

    fn try_from(value: Asn1Object) -> Result<Self> {
        if let Asn1Object::Tagged(tagged) = value {
            Self::try_from_base_object(tagged)
        } else {
            Err(BcError::with_invalid_argument("Expected a tagged object for GeneralName"))
        }
    }
}
impl TryFromTagged for GeneralName {
    fn try_from_tagged(tagged: Asn1TaggedObject, declared_explicit: bool) -> Result<Self>
    where
        Self: Sized,
    {
        try_from_choice_tagged(tagged, declared_explicit, GeneralName::try_from)
    }
}
