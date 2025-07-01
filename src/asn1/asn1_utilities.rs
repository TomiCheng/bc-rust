use crate::asn1::{asn1_tags, Asn1Object, Asn1Sequence};
use crate::asn1::asn1_universal_type::Asn1UniversalType;
use crate::Result;

pub(crate) fn read_optional_context_tagged<TResult, TMetadata>(
    sequence: &Asn1Sequence,
    sequence_position: usize,
    tag_no: u8,
    state: bool,
    metadata: TMetadata) -> Result<(Option<TResult>, usize)>
where TMetadata: Asn1UniversalType<TResult>
{
    read_optional_tagged(sequence, sequence_position, asn1_tags::CONTEXT_SPECIFIC, tag_no, state, metadata)
}

pub(crate) fn read_optional_tagged<TResult, TMetadata>(
    sequence: &Asn1Sequence,
    sequence_position: usize,
    tag_class: u8,
    tag_no: u8,
    state: bool,
    metadata: TMetadata)-> Result<(Option<TResult>, usize)>
where TMetadata: Asn1UniversalType<TResult>
{
    if sequence_position < sequence.len() {
        let result = try_get_optional_tagged(&sequence[sequence_position], tag_class, tag_no, state, metadata)?;
        if result.is_some() {
            let new_position = sequence_position + 1;
            return Ok((result, new_position));
        }
    }
    Ok((None, sequence_position))
}

//
pub(crate) fn try_get_optional_tagged<TResult, TMetadata>(
     element: &Asn1Object,
     tag_class: u8,
     tag_no: u8,
     state: bool, metadata: TMetadata) -> Result<Option<TResult>>
where TMetadata: Asn1UniversalType<TResult>
{
    if let Some(tagged_object) = element.as_tagged() {
        if tagged_object.has_tag(tag_class, tag_no) {
            let result = metadata.get_tagged(tagged_object, state)?;
            return Ok(Some(result));
        }
    }
    Ok(None)
}

