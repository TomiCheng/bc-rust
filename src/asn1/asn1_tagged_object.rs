use crate::asn1::asn1_tags::{PRIVATE, UNIVERSAL};
use crate::asn1::asn1_universal_type::Asn1UniversalType;
use crate::asn1::{Asn1EncodableVector, Asn1Object, Asn1OctetString, Asn1Sequence, asn1_tags, EncodingType};
use crate::{BcError, Result};
use crate::asn1::asn1_encodable::Asn1EncodingInternal;
use crate::asn1::asn1_encoding::Asn1Encoding;
use crate::asn1::tagged_der_encoding::TaggedDerEncoding;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Asn1TaggedObject {
    explicitness: u8,
    tag_class: u8,
    tag_no: u8,
    object: Box<Asn1Object>,
}
impl Asn1TaggedObject {
    const DECLARED_EXPLICIT: u8 = 1;
    const DECLARED_IMPLICIT: u8 = 2;
    const PARSED_EXPLICIT: u8 = 3;
    const PARSED_IMPLICIT: u8 = 4;
    pub(crate) fn new(explicitness: u8, tag_class: u8, tag_no: u8, object: Asn1Object) -> Result<Self> {
        if tag_class == UNIVERSAL || (tag_class & PRIVATE) != tag_class {
            return Err(BcError::with_invalid_argument(format!("invalid tag class: {}", tag_class)));
        }
        Ok(Self {
            explicitness,
            tag_class,
            tag_no,
            object: Box::new(object),
        })
    }
    pub(crate) fn from_explicit_tag_object(is_explicit: bool, tag_no: u8, asn1_object: Asn1Object) -> Self {
        let explicitness = if is_explicit { Self::DECLARED_EXPLICIT } else { Self::DECLARED_IMPLICIT };
        Self {
            explicitness,
            tag_class: asn1_tags::CONTEXT_SPECIFIC,
            tag_no,
            object: Box::new(asn1_object),
        }
    }
    // pub(crate) fn with_context_specific(explicitness: bool, tag_no: u8, object: Asn1Object) -> Result<Self> {
    //     Self::new(
    //         if explicitness {
    //             Self::DECLARED_EXPLICIT
    //         } else {
    //             Self::DECLARED_IMPLICIT
    //         },
    //         asn1_tags::CONTEXT_SPECIFIC,
    //         tag_no,
    //         object,
    //     )
    // }
    pub(crate) fn create_primitive(tag_class: u8, tag_no: u8, contents_octets: &[u8]) -> Result<Self> {
        Asn1TaggedObject::new(
            Self::PARSED_IMPLICIT,
            tag_class,
            tag_no,
            Asn1Object::from(Asn1OctetString::with_contents(contents_octets)),
        )
    }
    pub(crate) fn crate_constructed_dl(tag_class: u8, tag_no: u8, vector: Asn1EncodableVector) -> Result<Asn1Object> {
        let maybe_explicit = vector.len() == 1;
        if maybe_explicit {
            Ok(Asn1Object::from(Asn1TaggedObject::new(
                Self::PARSED_EXPLICIT,
                tag_class,
                tag_no,
                vector[0].clone(),
            )?))
        } else {
            let sequence = Asn1Object::from(Asn1Sequence::from_vector(vector)?);
            Ok(Asn1Object::from(Asn1TaggedObject::new(
                Self::PARSED_IMPLICIT,
                tag_class,
                tag_no,
                sequence,
            )?))
        }
    }
    pub fn has_tag(&self, tag_class: u8, tag_no: u8) -> bool {
        self.tag_class == tag_class && self.tag_no == tag_no
    }
    pub fn has_tag_class(&self, tag_class: u8) -> bool {
        self.tag_class == tag_class
    }
    pub(crate) fn has_context_tag(&self) -> bool {
        self.tag_class == asn1_tags::CONTEXT_SPECIFIC
    }
    pub fn is_explicit(&self) -> bool {
        self.explicitness == Self::DECLARED_EXPLICIT || self.explicitness == Self::PARSED_EXPLICIT
    }
    pub fn is_parsed(&self) -> bool {
        self.explicitness == Self::PARSED_EXPLICIT || self.explicitness == Self::PARSED_IMPLICIT
    }
    pub fn get_base_object(&self) -> &Asn1Object {
        &self.object
    }
    pub fn tag_no(&self) -> u8 {
        self.tag_no
    }
    pub fn tag_class(&self) -> u8 {
        self.tag_class
    }
    pub fn try_into_explicit_base_object(self) -> Result<Asn1Object> {
        if !self.is_explicit() {
            return Err(BcError::with_invalid_operation("object implicit - explicit expected."));
        }
        Ok(*self.object)
    }

    //

    pub(crate) fn try_from_base_universal<TMetadata, TResult>(self, declared_explicit: bool, metadata: TMetadata) -> Result<TResult>
    where
        TMetadata: Asn1UniversalType<TResult>,
    {
        if declared_explicit {
            if !self.is_explicit() {
                return Err(BcError::with_invalid_operation("object implicit - explicit expected."));
            }
            return metadata.checked_cast(*self.object);
        }

        if self.explicitness == Self::DECLARED_EXPLICIT {
            return Err(BcError::with_invalid_operation("object explicit - implicit expected."));
        }

        match self.explicitness {
            Self::PARSED_EXPLICIT => metadata.implicit_constructed(self.rebuild_constructed()),
            Self::PARSED_IMPLICIT => {
                if let Asn1Object::Sequence(sequence) = *self.object {
                    metadata.implicit_constructed(sequence)
                } else {
                    metadata.implicit_primitive((*self.object).try_into()?)
                }
            }
            _ => metadata.checked_cast(*self.object),
        }
    }
    fn rebuild_constructed(self) -> Asn1Sequence {
        Asn1Sequence::new(vec![*self.object])
    }
}
impl Asn1EncodingInternal for Asn1TaggedObject {
    fn get_encoding(&self, encoding_type: EncodingType) -> Box<dyn Asn1Encoding> {
        let base_object = self.get_base_object();

        if !self.is_explicit() {
            return base_object.get_encoding_implicit(encoding_type, self.tag_class, self.tag_no);
        }

        match encoding_type {
            EncodingType::Der => Box::new(TaggedDerEncoding::new(self.tag_class, self.tag_no, base_object.get_encoding(encoding_type))),
            EncodingType::Ber => Box::new(TaggedDerEncoding::new(self.tag_class, self.tag_no, base_object.get_encoding(encoding_type))),
            EncodingType::Dl => Box::new(TaggedDerEncoding::new(self.tag_class, self.tag_no, base_object.get_encoding(encoding_type))),
        }
    }

    fn get_encoding_implicit(&self, encoding_type: EncodingType, tag_class: u8, tag_no: u8) -> Box<dyn Asn1Encoding> {
        let base_object = self.get_base_object();
        if !self.is_explicit() {
            return base_object.get_encoding_implicit(encoding_type, tag_class, tag_no);
        }
        Box::new(TaggedDerEncoding::new(tag_class, tag_no, base_object.get_encoding(encoding_type)))
    }
}