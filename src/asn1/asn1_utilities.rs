use crate::asn1::{Asn1Object, Asn1TaggedObject, asn1_tags};
use crate::{BcError, Result};
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

pub(crate) fn try_from_choice_tagged<TResult, TFunc>(tagged: Asn1TaggedObject, declared_explicit: bool, func: TFunc) -> Result<TResult>
where
    TFunc: FnOnce(Asn1Object) -> Result<TResult>,
{
    if !declared_explicit {
        return Err(BcError::with_invalid_argument(
            "Implicit tagging cannot be used with untagged choice type (X.680 30.6, 30.8).",
        ));
    }
    func(try_from_explicit_context_base_object(tagged)?)
}

pub(crate) fn try_from_explicit_context_base_object(tagged: Asn1TaggedObject) -> Result<Asn1Object> {
    try_from_explicit_base_object(tagged, asn1_tags::CONTEXT_SPECIFIC)
}

pub(crate) fn try_from_explicit_base_object(tagged: Asn1TaggedObject, tag_class: u8) -> Result<Asn1Object> {
    check_tag_class(tagged, tag_class)?.try_into_explicit_base_object()
}

pub(crate) fn check_tag_class(tagged: Asn1TaggedObject, tag_class: u8) -> Result<Asn1TaggedObject> {
    if !tagged.has_tag_class(tag_class) {
        let expected = get_tag_class_text_from_class_no(tag_class);
        let found = get_tag_class_text_from_tagged(&tagged);
        return Err(BcError::with_invalid_operation(format!("expected {}, tag but found {}", expected, found)));
    }
    Ok(tagged)
}

fn get_tag_class_text_from_tagged(tagged: &Asn1TaggedObject) -> String {
    get_tag_class_text_from_class_no(tagged.tag_class())
}

fn get_tag_class_text_from_class_no(tag_class: u8) -> String {
    match tag_class {
        asn1_tags::UNIVERSAL => "UNIVERSAL".to_string(),
        asn1_tags::APPLICATION => "APPLICATION".to_string(),
        asn1_tags::CONTEXT_SPECIFIC => "CONTEXT".to_string(),
        asn1_tags::PRIVATE => "PRIVATE".to_string(),
        _ => format!("UNKNOWN({})", tag_class),
    }
}
