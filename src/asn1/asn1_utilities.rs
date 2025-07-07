use crate::Result;
use crate::asn1::{Asn1Object, Asn1TaggedObject, asn1_tags};
use std::iter::Peekable;
use std::vec::IntoIter;

// pub(crate) fn read_optional_context_tagged<TResult, TMetadata>(
//     sequence: &Asn1Sequence,
//     sequence_position: usize,
//     tag_no: u8,
//     state: bool,
//     metadata: TMetadata,
// ) -> Result<(Option<TResult>, usize)>
// where
//     TMetadata: Asn1UniversalType<TResult>,
// {
//     read_optional_tagged(sequence, sequence_position, asn1_tags::CONTEXT_SPECIFIC, tag_no, state, metadata)
// }
// 
// pub(crate) fn read_optional_tagged<TResult, TMetadata>(
//     sequence: &Asn1Sequence,
//     sequence_position: usize,
//     tag_class: u8,
//     tag_no: u8,
//     state: bool,
//     metadata: TMetadata,
// ) -> Result<(Option<TResult>, usize)>
// where
//     TMetadata: Asn1UniversalType<TResult>,
// {
//     if sequence_position < sequence.len() {
//         let result = try_get_optional_tagged(&sequence[sequence_position], tag_class, tag_no, state, metadata)?;
//         if result.is_some() {
//             let new_position = sequence_position + 1;
//             return Ok((result, new_position));
//         }
//     }
//     Ok((None, sequence_position))
// }
// 
// //
// pub(crate) fn try_get_optional_tagged<TResult, TMetadata>(
//     element: &Asn1Object,
//     tag_class: u8,
//     tag_no: u8,
//     state: bool,
//     metadata: TMetadata,
// ) -> Result<Option<TResult>>
// where
//     TMetadata: Asn1UniversalType<TResult>,
// {
//     if let Some(tagged_object) = element.as_tagged() {
//         if tagged_object.has_tag(tag_class, tag_no) {
//             let result = metadata.get_tagged(tagged_object, state)?;
//             return Ok(Some(result));
//         }
//     }
//     Ok(None)
// }

pub(crate) fn read_optional_context_iter<TResult, TState, TFunc>(
    iter: &mut Peekable<IntoIter<Asn1Object>>,
    tag_no: u8,
    state: TState,
    func: TFunc,
) -> Result<Option<TResult>>
where
    TFunc: Fn(Asn1TaggedObject, TState) -> Result<TResult>,
{
    if let Some(tagged_object) = iter.peek().and_then(Asn1Object::as_tagged) {
        if tagged_object.has_tag(asn1_tags::CONTEXT_SPECIFIC, tag_no) {
            let tagged: Asn1TaggedObject = iter.next().unwrap().try_into()?;
            return Ok(Some(func(tagged, state)?));
        }
    }
    Ok(None)
}

pub(crate) fn read_context_iter<TResult, TState, TFunc>(
    iter: &mut Peekable<IntoIter<Asn1Object>>,
    tag_no: u8,
    state: TState,
    func: TFunc,
) -> Result<TResult>
where
    TFunc: Fn(Asn1TaggedObject, TState) -> Result<TResult>,
{
    let tagged: Asn1TaggedObject = iter.next().unwrap().try_into()?;
    func(tagged, state)
}
