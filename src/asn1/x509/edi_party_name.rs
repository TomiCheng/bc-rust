use crate::asn1::Asn1Sequence;
use crate::{BcError, Result};
use crate::asn1::asn1_utilities::{read_context_iter, read_optional_context_iter};
use crate::asn1::x500::DirectoryString;

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
    fn new(name_assigner: Option<DirectoryString>, party_name: DirectoryString) -> EdiPartyName {
        EdiPartyName {
            name_assigner,
            party_name,
        }
    }
    pub fn name_assigner(&self) -> Option<&DirectoryString> {
        self.name_assigner.as_ref()
    }
    pub fn party_name(&self) -> &DirectoryString {
        &self.party_name
    }

    pub(crate) fn from_sequence(sequence: Asn1Sequence) -> Result<Self> {
        // if sequence.len() < 1 || sequence.len() > 2 {
        //     return Err(BcError::with_invalid_argument("EdiPartyName must have 1 or 2 elements"));
        // }
        // let mut iter = sequence.into_iter().peekable();
        // let name_assigner = read_optional_context_iter(&mut iter, 0, true, DirectoryString::get_tagged)?;
        // let party_name = read_context_iter(&mut iter, 1, true, DirectoryString::get_tagged)?;
        // Ok(EdiPartyName::new(name_assigner, party_name))
        
        todo!();
    }
}