use super::der_encoding::{DerEncoding, DerEncodingImpl};

pub(crate) struct TaggedDerEncoding {
    parent: DerEncodingImpl,
    contents_element: Box<dyn DerEncoding>,
    contents_length: usize,
}

impl TaggedDerEncoding {
    pub(crate) fn new(tag_class: u32, tag_no: u32, contents_element: Box<dyn DerEncoding>) -> Self {
        let length = contents_element.get_length(); 
        TaggedDerEncoding {
            parent: DerEncodingImpl::new(tag_class, tag_no),
            contents_element,
            contents_length: length,
        }
    }
    // pub fn get_length(&self) -> usize {
    //     get_length_of_encoding_dl(self.parent.get_tag_no(), self.contents_length)
    // }
    // fn write(&self, writer: &mut Asn1WriteImpl) -> std::io::Result<()> {
    //     writer.write_identifier(self.parent.get_tag_class(), self.parent.get_tag_no())?;
    //     writer.write_dl(self.contents_length as u32)?;
    //     self.contents_element.write(writer)?;
    //     return Ok(());
    // }
    // pub fn encode<T: Any + Asn1Write>(&self, writer: &mut T) -> Result<()> {
    //     let writer_any = writer as &mut dyn Any;
    //     if let Some(writer) = writer_any.downcast_mut::<Asn1WriteImpl>() {
    //         self.write(writer).map_err(|e| BcError::IoError {
    //             msg: "Failed to write DER encoding".to_string(),
    //             source: e,
    //         })?;
    //         return Ok(());
    //     } else {
    //         panic!("downcast_mut failed");
    //     }
    // }
    // pub(crate) fn compare_length_and_contents<T: Any>(&self, other: &T) -> Option<Ordering> {
    //     let other_any = other as &dyn Any;
    //     if let Some(other) = other_any.downcast_ref::<TaggedDerEncoding>() {
    //         return other
    //             .compare_length_and_contents(self)
    //             .map(|ordering| ordering.reverse());
    //     } else {
    //         return None;
    //     }
    // }
}