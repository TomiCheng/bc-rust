use crate::asn1::asn1_utilities::{read_context_iter, read_optional_context_iter};
use crate::asn1::try_from_tagged::{TryFromTagged, TryIntoTagged};
use crate::asn1::x500::DirectoryString;
use crate::asn1::{Asn1Object, Asn1Sequence, Asn1TaggedObject};
use crate::{BcError, Result};

/// ```text
/// EdiPartyName ::= SEQUENCE {
///     nameAssigner [0] Name OPTIONAL,
///     partyName [1] Name
/// }
/// ```
pub struct EdiPartyName {
    name_assigner: Option<DirectoryString>,
    party_name: DirectoryString,
}

impl EdiPartyName {
    pub fn new(name_assigner: Option<DirectoryString>, party_name: DirectoryString) -> EdiPartyName {
        EdiPartyName { name_assigner, party_name }
    }
    pub fn name_assigner(&self) -> Option<&DirectoryString> {
        self.name_assigner.as_ref()
    }
    pub fn party_name(&self) -> &DirectoryString {
        &self.party_name
    }
    pub(crate) fn from_sequence(sequence: Asn1Sequence) -> Result<Self> {
        if sequence.len() < 1 || sequence.len() > 2 {
            return Err(BcError::with_invalid_argument("EdiPartyName must have 1 or 2 elements"));
        }
        let mut iter = sequence.into_iter().peekable();

        let name_assigner = read_optional_context_iter(&mut iter, 0, true, DirectoryString::try_from_tagged)?;
        let party_name = read_context_iter(&mut iter, 1, true, DirectoryString::try_from_tagged)?;
        Ok(EdiPartyName::new(name_assigner, party_name))
    }
}
impl From<EdiPartyName> for Asn1Object {
    fn from(value: EdiPartyName) -> Self {
        if let Some(name_assigner) = value.name_assigner {
            Asn1Sequence::new(vec![name_assigner.into(), value.party_name.into()]).into()
        } else {
            Asn1Sequence::new(vec![value.party_name.into()]).into()
        }
    }
}
impl TryFrom<Asn1Object> for EdiPartyName {
    type Error = BcError;
    fn try_from(value: Asn1Object) -> Result<Self> {
        Self::from_sequence(value.try_into()?)
    }
}
impl TryFromTagged for EdiPartyName {
    fn try_from_tagged(tagged: Asn1TaggedObject, declared_explicit: bool) -> Result<Self>
    where
        Self: Sized,
    {
        Self::from_sequence(tagged.try_into_tagged(declared_explicit)?)
    }
}
