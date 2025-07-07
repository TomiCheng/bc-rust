use crate::{BcError, Result};
use crate::asn1::{Asn1Convertible, Asn1Ia5String, Asn1Object, Asn1ObjectIdentifier, Asn1OctetString, Asn1Sequence, Asn1TaggedObject};
use crate::asn1::x509::{EdiPartyName, X509Name};

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
    pub(crate) fn from_asn1_object(asn1_object: Asn1Object) -> Result<Self> {
        if let Asn1Object::Tagged(tagged_object) = asn1_object {
            Self::get_optional_base_object(tagged_object)
        } else {
            Err(BcError::with_invalid_argument("Expected a tagged object for GeneralName"))
        }
    }
    fn get_optional_base_object(tagged_object: Asn1TaggedObject) -> Result<Self> {
        if tagged_object.has_context_tag() {
            match tagged_object.tag_no() {
                Self::OTHER_NAME => Ok(GeneralName::OtherName(Asn1Sequence::get_tagged(tagged_object, false)?)),
                Self::RFC822_NAME => Ok(GeneralName::Rfc822Name(Asn1Ia5String::get_tagged(tagged_object, false)?)),
                Self::DNS_NAME => Ok(GeneralName::DnsName(Asn1Ia5String::get_tagged(tagged_object, false)?)),
                Self::X400_ADDRESS => Ok(GeneralName::X400Address(Asn1Sequence::get_tagged(tagged_object, false)?)),
                Self::DIRECTORY_NAME => Ok(GeneralName::DirectoryName(X509Name::get_tagged(tagged_object, false)?)),
                Self::EDI_PARTY_NAME => {
                    let sequence = Asn1Sequence::get_tagged(tagged_object, false)?;
                    Ok(GeneralName::EdiPartyName(EdiPartyName::from_sequence(sequence)?))
                },
                Self::UNIFORM_RESOURCE_IDENTIFIER => Ok(GeneralName::UniformResourceIdentifier(Asn1Ia5String::get_tagged(tagged_object, false)?)),
                Self::IP_ADDRESS => Ok(GeneralName::IpAddress(Asn1OctetString::get_tagged(tagged_object, false)?)),
                Self::REGISTERED_ID => Ok(GeneralName::RegisteredId(Asn1ObjectIdentifier::get_tagged(tagged_object, false)?)),
                _ => Err(BcError::with_invalid_argument(format!("Expected a context tag for GeneralName tag_no: {}", tagged_object.tag_no()))),
            }
        } else {
            Err(BcError::with_invalid_argument("Expected a context tag for GeneralName"))
        }
    }
}

impl Asn1Convertible for GeneralName {
    fn to_asn1_object(&self) -> Result<Asn1Object> {
        //let is_explicit = self.tag_no == Self::DIRECTORY_NAME;
        //Ok(Asn1TaggedObject::with_context_specific(is_explicit, self.tag_no, self.object.clone())?.into())
        todo!();
    }
}

