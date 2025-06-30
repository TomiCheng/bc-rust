use crate::asn1::{asn1_tags, Asn1Object, Asn1Sequence, Asn1TaggedObject};
use crate::Result;

pub fn read_optional_context_tagged<TState, TResult, F>(sequence: &Asn1Sequence, 
                                                        sequence_position: usize,  
                                                        tag_no: u8, 
                                                        state: TState, 
                                                        constructor: F) -> Result<(Option<TResult>, usize)>
where F: FnOnce(&Asn1TaggedObject, TState) -> Result<TResult>
{
    read_optional_tagged(sequence, sequence_position, asn1_tags::CONTEXT_SPECIFIC, tag_no, state, constructor)
}

pub fn read_optional_tagged<TState, TResult, F>(
    sequence: &Asn1Sequence, 
    sequence_position: usize, 
    tag_class: u8, 
    tag_no: u8, 
    state: TState, 
    constructor: F) -> Result<(Option<TResult>, usize)>
where F: FnOnce(&Asn1TaggedObject, TState) -> Result<TResult>
{
    if sequence_position < sequence.len() {
        let result = try_get_optional_tagged(&sequence[sequence_position], tag_class, tag_no, state, constructor)?;
        if result.is_none() {
            let sequence_position = sequence_position + 1;
            return Ok((result, sequence_position));
        }
    }
    Ok((None, sequence_position))
}

pub fn try_get_optional_tagged<TState, TResult, F>(
    element: &Asn1Object, 
    tag_class: u8, 
    tag_no: u8, 
    state: TState, func: F) -> Result<Option<TResult>>
where F: FnOnce(&Asn1TaggedObject, TState) -> Result<TResult>
{
    if let Some(tagged_object) = element.as_tagged() {
        if tagged_object.has_tag(tag_class, tag_no) {
            let result =  func(&tagged_object, state)?;
            return Ok(Some(result));
        }
    }
    Ok(None)
}

